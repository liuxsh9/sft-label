# OpenAI-style Turn Compatibility 设计

## 背景

当前 `/Users/lxs/.codex/worktrees/e4c5/sft-label` 的 Pass 1 入口支持两类主格式：

1. ShareGPT 风格：`conversations[].from/value`
2. Pangu 风格：`data[].role/content`

目录扫描本身已经支持递归读取 `.json` / `.jsonl` 文件，因此像 `/Volumes/MOVESPEED/datasets/Step-3.5-Flash-SFT-code-sft-jsonl-qwen3-coder-plus` 这样的目录可以被发现并进入读取流程。但该目录中的行数据主要是：

```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

现有 `normalize_and_slice()` 在顶层看到 `conversations` 后，会直接按 ShareGPT 路径处理；然而后续多轮切片逻辑直接读取 `turn["from"]`，导致 `conversations[].role/content` 在真实运行时触发 `KeyError: 'from'`。

这不是单轮样本的孤立问题。对目标目录前 2000 行抽样后，已经确认存在：

- 大量单轮 `user/assistant`
- 普通多轮 `user/assistant/user/assistant`
- 含 `system`
- 含 `tool`
- 长轨迹、多个 assistant 回复的 agent/tool-use 样本

因此，本次设计不能只做“字段改名补丁”，而必须让流水线原生支持一类更广义的 **OpenAI-style turn schema**，并确保单轮、多轮、system/tool 轨迹都可以稳定进入现有标注与切片流程。

## 目标

1. 原生支持以下 turn schema：
   - `conversations[].from/value`
   - `conversations[].role/content`
   - `messages[].role/content`
2. 将上述 schema 统一标准化为内部格式：`from/value` + 规范化角色（`human/gpt/tool/system`）。
3. 保持当前多轮切片语义不变：每个 assistant 回复生成一个训练样本，保留全部前文。
4. 确保 `system` 与 `tool` turn 在标准化与切片后不丢失。
5. 保证现有 ShareGPT 与 Pangu 路径不回归。
6. 为真实大目录抽样验证提供稳定入口，确认目标目录中的单轮 / 多轮 / 长轨迹样本都能正常切片和进入后续标注流程。

## 非目标

1. 不修改现有 taxonomy、Pass 1 prompt、Pass 2 scoring 语义。
2. 不在 pipeline / scoring / dashboard 中散布新的 schema 分支判断。
3. 不把任意未知 turn 字段自动猜测为合法格式；仅支持明确声明的兼容 schema。
4. 不在本次改动中新增新的数据源持久化格式。
5. 不尝试一次性支持所有聊天数据标准（例如 Anthropic 自定义消息块、OpenAI 多模态 content blocks 等复杂变体）。

## 核心原则

### 1. 入口标准化，后续复用现有流程

所有新增兼容性都应尽量收敛在 `/Users/lxs/.codex/worktrees/e4c5/sft-label/src/sft_label/preprocessing.py` 的入口标准化层。标准化完成后，后续：

- `slice_multiturn()`
- thinking/COT 提取
- trajectory object 检测
- inline row ingestion
- Pass 1 / Pass 2

都继续只依赖现有内部格式，不新增额外 schema 条件分支。

### 2. 多轮语义优先，不能为了兼容字段牺牲切片正确性

支持 `role/content` 的目的，不只是“避免报错”，而是要保证：

- assistant 回复数与输出 sample 数一致；
- 每个 slice 保留完整前文；
- `turn_index` / `total_turns` / `conversation_uid` 等元信息保持一致；
- `system`、`tool`、长轨迹行为不被削弱。

### 3. 兼容层必须幂等

对于已经是 `from/value` 的样本，标准化层应尽量保持原结构与语义不变。不能为了支持新 schema 而改变现有 ShareGPT 样本的字段内容或 metadata 语义。

### 4. 兼容范围显式、优先级固定

顶层格式识别与兼容优先级需要固定，避免同一条样本被多重解释：

1. `data` → Pangu
2. `conversations` → conversation-based schema
3. `messages` → OpenAI-style schema
4. 否则 `unknown`

若样本同时包含 `conversations` 与 `messages`，优先使用 `conversations`，避免歧义。

---

## 总体方案

```text
raw row
  -> detect top-level format
  -> normalize turns to internal conversations[]
       turn shape: {from, value}
       role map: user/human -> human
                 assistant/gpt -> gpt
                 tool -> tool
                 system -> system
  -> existing normalize_and_slice flow
  -> existing slice_multiturn / metadata / downstream pipeline
```

### 方案选择

本次选择 **统一的 OpenAI-style turn 兼容层**，而不是：

- 只针对 `conversations[].role/content` 做一次性字段补丁；或
- 在 pipeline 各层分别加条件分支。

原因：

1. 目标目录已证明确实存在真实多轮、system、tool 长轨迹，不能靠最小补丁赌运气；
2. `messages[].role/content` 是同一类问题，顺手统一兼容层成本低、收益高；
3. 标准化层单点收敛后，测试也更容易系统化覆盖。

---

## 输入兼容模型

### 支持的顶层格式

#### A. 现有 ShareGPT 风格

```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

行为：保持现状，标准化应基本幂等。

#### B. 新增 conversations + role/content 风格

```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

行为：转成内部 `from/value`，再走现有切片。

#### C. 新增 messages + role/content 风格

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

行为：先映射为 `conversations`，再走同一套标准化。

#### D. 现有 Pangu 风格

```json
{
  "data": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

行为：继续走 `normalize_pangu()`；其输出已经是内部 `from/value`。

### Turn 级角色映射

标准角色映射为：

- `user` / `human` → `human`
- `assistant` / `gpt` → `gpt`
- `tool` → `tool`
- `system` → `system`

对已经合法的内部角色保持不变。

### Turn 文本字段映射

- `value` 优先保留为内部字段
- 若不存在 `value`，则读取 `content`
- 若两者都不存在，则回退为空字符串 `""`

### 未知角色策略

未知角色不进行“智能猜测”。保留其标准化后的原小写 role 值，使现有 `_canonical_role()`、轨迹检测和下游分析维持当前宽松行为，而不是静默丢弃 turn。

---

## 预处理设计

### 新增统一 turn 标准化 helper

建议在 `/Users/lxs/.codex/worktrees/e4c5/sft-label/src/sft_label/preprocessing.py` 中新增清晰的 helper，例如：

- `_normalize_turn_role(role)`
- `_normalize_turn_record(turn)`
- `_normalize_openai_style_conversations(turns)`
- `_normalize_openai_style_messages(sample)`

职责划分：

1. **role 标准化**：统一大小写和别名映射。
2. **turn 标准化**：输出 `{from, value}`，保留原 turn 顺序。
3. **top-level 标准化**：
   - `conversations` 若已经是 `from/value` 列表，则幂等通过；
   - `conversations` 若为 `role/content` 列表，则转换；
   - `messages` 若存在，则转换为 `conversations`。

### `detect_format()` 调整

当前 `detect_format()` 对 `messages` 返回 `unknown`。本次应调整为显式识别 OpenAI-style，例如返回 `openai_messages` 或继续使用 `sharegpt` 但通过 helper 判断。关键是：

- `data` 优先视为 Pangu；
- `conversations` 与 `messages` 都能进入标准化流程；
- `unknown` 仅留给真正未支持的结构。

为了减小下游改动，推荐：

- 保持 `pangu` 分支不动；
- 将 `conversations` / `messages` 两类都归入同一非-Pangu标准化分支。

### `normalize_and_slice()` 调整

当前非 Pangu 路径直接 `normalized = dict(sample)`，并默认 `conversations` 已是内部结构。本次应改为：

1. 对 `conversations` / `messages` 先做 top-level turn 标准化；
2. 之后再执行 thinking/COT 提取与 strip；
3. 再进入 `slice_multiturn()`。

这样可以保证 `slice_multiturn()` 永远面对标准的 `{from, value}` turn，而不再依赖外部 schema 是否碰巧匹配。

### 多轮切片逻辑保持不变

`slice_multiturn()` 仍只根据最终标准化后的 `from == "gpt"` 来定位 assistant 回复。对于真实样本：

- 单轮 `user/assistant` → 1 个 sample
- `user, assistant, user, assistant` → 2 个 sample
- `system, user, assistant, tool, assistant` → 2 个 sample
- 长轨迹（多个 assistant） → N 个 sample

即：**assistant 回复数 = 输出切片数**。

---

## 下游影响评估

### inline row ingestion

`/Users/lxs/.codex/worktrees/e4c5/sft-label/src/sft_label/inline_rows.py` 的 `build_row_sample_bundle()` 直接调用 `normalize_and_slice()`。因此只要预处理入口标准化正确：

- JSONL 文件读取
- row bundle 构建
- inline Pass 1 merge

都能自动继承支持，不需要额外 schema 分支。

### pipeline 目录模式

目录扫描和逐文件读取已经能发现目标目录中的 100 个 JSONL 文件；实际失败点仅在预处理。因此修复入口标准化后，目录模式可直接受益。

### Pass 2 / conversation / dashboard

这些阶段消费的是 Pass 1 之后的标准化样本，不应感知新增 schema。只要 Pass 1 输出样本结构不变，后续不需要功能性改动。

---

## 测试策略

本次测试必须覆盖“字段兼容”和“真实多轮语义”两个维度，分 3 层实施。

### 第 1 层：预处理单测

文件：`/Users/lxs/.codex/worktrees/e4c5/sft-label/tests/test_preprocessing.py`

新增/扩展 case：

1. `conversations[].role/content` 单轮
   - `user -> assistant`
   - 断言输出 1 个 sample
   - 断言 turn 标准化为 `human/gpt`

2. `conversations[].role/content` 双 assistant 多轮
   - `user, assistant, user, assistant`
   - 断言输出 2 个 sample
   - 断言第二个 slice 保留全部前文

3. `conversations[].role/content` + `system`
   - 断言 leading `system` 被保留在每个 slice 中

4. `conversations[].role/content` + `tool`
   - 例如 `system, user, assistant, tool, assistant`
   - 断言输出 2 个 sample 且 tool turn 存在于对应 slice 中

5. `messages[].role/content`
   - 单轮和多轮至少各 1 条
   - 断言最终行为与 `conversations[].role/content` 一致

6. 长轨迹样本
   - 至少 5 个 assistant 回复
   - 断言 sample 数、`turn_index`、`total_turns`、最后一个 slice 上下文正确

7. 旧格式回归
   - 现有 `from/value`
   - 现有 Pangu `data[].role/content`
   - 确认全部继续通过

### 第 2 层：行级 JSONL ingestion 单测

文件：`/Users/lxs/.codex/worktrees/e4c5/sft-label/tests/test_inline_pass1.py`

新增 case：

1. 临时 JSONL 文件中写入 `conversations[].role/content` 单轮行
2. 写入多轮行（2 个 assistant）
3. 写入含 `system/tool` 的长轨迹行
4. 同一 JSONL 文件中混合多种 schema（如一行 `messages`、一行 `conversations role/content`）

验证：

- `build_row_sample_bundle()` 不报错；
- `iter_row_sample_bundles_from_jsonl()` 能逐行产出 bundle；
- `flatten_row_sample_bundles()` 后 sample 数符合 assistant 回复数；
- stable sample id / turn metadata 依旧正确。

### 第 3 层：真实目录抽样验证

此层不依赖 CI，不把 `/Volumes/...` 硬编码进正式测试。采用本地验证脚本 / 命令方式，对目标目录抽样做兼容性检查。

抽样至少覆盖：

- 单轮样本
- 两个 assistant 的普通多轮
- 三个及以上 assistant 的长轨迹
- 含 `tool`
- 含 `system`

验证项：

- 不报错；
- assistant 回复数 = 输出 sample 数；
- turn 角色被正确标准化；
- 长轨迹中 `system/tool` 未丢失。

---

## 风险与缓解

### 风险 1：现有 ShareGPT 样本回归

**表现**：已使用 `from/value` 的旧样本被重复改写或 metadata 语义改变。

**缓解**：

- 标准化 helper 设计成幂等；
- 为旧 ShareGPT 样本保留回归测试；
- focused test 先覆盖老 case，再扩新 case。

### 风险 2：多轮 sample 数错误

**表现**：新增兼容后，只能避免 crash，但切片数不再等于 assistant 回复数。

**缓解**：

- 多轮测试显式断言 sample 数；
- 长轨迹测试显式断言 `turn_index` / `total_turns`；
- 对第二个/最后一个 slice 断言前文完整保留。

### 风险 3：`system` / `tool` 被静默丢弃

**表现**：切片可运行，但上下文质量退化，影响 agent/tool-use 数据标注。

**缓解**：

- 单独增加含 `system` / `tool` 的测试；
- 对真实样本抽样验证 role 顺序；
- 保持 turn 顺序不变，只做字段标准化。

### 风险 4：`messages` 与 `conversations` 双字段歧义

**表现**：同一条样本被错误选择输入源。

**缓解**：

- 固定优先级：`data > conversations > messages`；
- 测试显式覆盖该优先级；
- 文档化该行为，避免未来误解。

---

## 实施步骤

1. 在 `tests/test_preprocessing.py` 中先写失败测试，覆盖：
   - `conversations[].role/content`
   - `messages[].role/content`
   - 单轮 / 多轮 / system / tool / 长轨迹
2. 运行 focused tests，确认当前失败点真实存在。
3. 在 `preprocessing.py` 中实现统一 turn 标准化 helper，并接入 `normalize_and_slice()`。
4. 让 `detect_format()`/入口分支正确识别 `messages` 与 `conversations role/content`。
5. 在 `tests/test_inline_pass1.py` 中补行级 JSONL ingestion 回归测试。
6. 运行 focused verification：
   - `uv run pytest tests/test_preprocessing.py -q`
   - `uv run pytest tests/test_inline_pass1.py -q`
   - 若 sample counting / estimation 受影响，再补相应 focused tests。
7. 在本地对目标目录做抽样探测，确认真实单轮 / 多轮 / system/tool 长轨迹都能正常切片。

---

## 完成定义

满足以下条件时，本次改动视为完成：

1. `sft-label` 能直接读取 `/Volumes/MOVESPEED/datasets/Step-3.5-Flash-SFT-code-sft-jsonl-qwen3-coder-plus` 这类目录，而不会在预处理阶段因 `role/content` turn 结构报错。
2. 对抽样真实数据，以下样本均能正常标准化和切片：
   - 单轮
   - 普通多轮
   - 含 `system`
   - 含 `tool`
   - 长轨迹、多 assistant 回复
3. 现有 `conversations[].from/value` 与 Pangu `data[].role/content` 测试继续通过。
4. 新增回归测试能稳定保护上述兼容行为，防止未来再出现“目录支持但 schema 不支持”的回归。
