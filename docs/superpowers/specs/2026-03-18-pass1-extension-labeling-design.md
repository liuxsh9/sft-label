# Pass1 Extension Labeling 设计

## 背景

当前 `/Users/lxs/.codex/worktrees/e8fb/sft-label` 的 Pass1 标注流程固定为两次 LLM 调用：

1. Call1：`intent` / `language` / `domain` / `task` / `difficulty`
2. Call2：`concept` / `agentic` / `constraint` / `context`

随后系统会执行标签校验、一致性检查、可选仲裁，并产出：

- `labeled.json` / inline `data_label`
- `stats_labeling.json`
- Pass1 dashboard

新的需求是：允许用户在 Pass1 阶段为特定领域的数据追加更细粒度的自定义标签。典型场景是 UI 数据：用户希望提供一个自定义 prompt，并声明一组明确字段 schema，让系统自动完成细化标注、持久化和 dashboard 展示。

这个能力不能只服务于 UI。系统应从一开始就支持多个 extension 配置并存，避免未来将不同垂域需求不断硬编码进核心 taxonomy。

## 目标

1. 在 Pass1 核心两次调用之后，增加一个可插拔的 **Extension Labeling Layer**。
2. 支持用户同时加载多个 extension 配置文件。
3. 每个 extension 由 `prompt + 明确字段 schema + trigger` 驱动。
4. 扩展标签与核心 9 维 taxonomy 隔离存储，不污染现有 `labels` 语义。
5. dashboard 支持对 extension 结果进行独立统计与展示。
6. 支持标准 run、inline run、recompute、regenerate-dashboard、交互式 `start` 路径。
7. 在实现上为多 agent 并行开发留出清晰边界。

## 非目标

1. 不将 extension 标签直接纳入核心 `TAG_POOLS`。
2. 不默认让 extension 标签参与 Pass2 rarity、selection 或 scoring prompt。
3. v1 不支持自由文本字段、嵌套对象 schema、任意 JSON schema。
4. v1 不引入后端服务或数据库；仍沿用静态文件与 inline JSONL 结构。
5. v1 不做 extension 级别的仲裁重跑，除非后续验证有明确必要性。

## 核心原则

### 1. 核心 taxonomy 稳定

现有 9 维 taxonomy 在 pipeline、stats、dashboard、scoring 中被广泛假定为固定结构。extension 必须作为平行层存在，而不是“第 10 个 taxonomy 维度”。

### 2. extension 非阻断

若 core Pass1 成功而某个 extension 调用失败：

- 样本整体仍视为 Pass1 成功；
- core `labels` 保持可用；
- extension 记录失败状态；
- 不把样本标成 partial core label。

### 3. schema 驱动而非自由输出

只有明确字段 schema，才能保证：

- 可校验
- 可聚合统计
- 可稳定渲染 dashboard
- 可支持 review/export/explorer

### 4. 配置版本可追踪

extension 结果必须记录 `spec_version` 与 `spec_hash`。用户修改 prompt 或 schema 后，系统需要知道旧结果已经过期，不能把不同配置跑出的结果误当作同一种标签。

---

## 总体架构

```text
Pass1 Core
  preprocess
  -> call1
  -> call2
  -> validate core labels
  -> merge core labels

Pass1 Extension Layer
  -> load enabled extension specs
  -> route by trigger
  -> run extension call(s)
  -> validate against extension schema
  -> persist label_extensions

Downstream
  -> extension stats
  -> extension dashboard sections
  -> inline persistence
  -> optional review/export exposure
```

### 推荐挂载点

在 `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/pipeline.py` 的 `label_one()` 中：

- core Call2 结果已校验并合并后；
- 进入 consistency / low-confidence arbitration 之后或紧接 core merge 之后；
- extension 使用稳定的 core labels 作为触发条件与上下文输入。

推荐顺序：

1. 完成 core merge
2. 完成 core consistency / low-confidence 处理
3. 运行 extension router
4. 返回 `labels + label_extensions + monitor`

这样可以确保 extension 看到的是最终 core labels，而不是半成品。

---

## Extension Spec 设计

v1 使用 YAML/JSON 配置文件，支持多个并存。

### 示例

```yaml
id: ui_fine_labels
spec_version: v1
display_name: UI Fine Labels
enabled: true

description: Fine-grained labels for UI-heavy frontend samples.

trigger:
  domain_any_of: [web-frontend]
  language_any_of: [html, css, javascript, typescript]

prompt: |
  你是一个 UI 数据细粒度标注器。
  请基于 conversation、preprocessed signals、以及 core labels，
  严格按照 schema 输出。

schema:
  component_type:
    type: multi_enum
    options: [form, table, modal, chart, nav, card, editor]
    description: Main UI components present in the task or response.

  interaction_pattern:
    type: multi_enum
    options: [crud, wizard, search-filter, drag-drop, inline-editing, dashboard]

  visual_complexity:
    type: enum
    options: [low, medium, high]

  ui_framework:
    type: multi_enum
    options: [react, vue, angular, nextjs, tailwind, antd, mui]

output:
  include_confidence: true
  allow_unmapped: true

dashboard:
  group: ui
  priority: 10
```

## v1 支持字段类型

- `enum`
- `multi_enum`

两者都要求 `options` 为稳定 ID 列表。

## v1 触发条件

建议支持以下简单组合：

- `domain_any_of`
- `domain_all_of`
- `language_any_of`
- `intent_any_of`
- `task_any_of`
- `context_any_of`
- `difficulty_any_of`

语义：只有在 core labels 命中 trigger 时，才会执行对应 extension。

---

## 数据模型设计

## 样本级输出

保留现有 core：

```json
"labels": { ...现有9维... }
```

新增平行字段：

```json
"label_extensions": {
  "ui_fine_labels": {
    "status": "success",
    "spec_version": "v1",
    "spec_hash": "sha256:...",
    "matched": true,
    "labels": {
      "component_type": ["form", "table"],
      "interaction_pattern": ["crud"],
      "visual_complexity": "medium"
    },
    "confidence": {
      "component_type": 0.86,
      "interaction_pattern": 0.77,
      "visual_complexity": 0.71
    },
    "unmapped": [],
    "monitor": {
      "llm_calls": 1,
      "elapsed_seconds": 0.9,
      "status": "success"
    }
  }
}
```

### status 建议值

- `success`
- `skipped`（未命中 trigger）
- `failed`
- `invalid`（LLM 返回结果不符合 schema）

### 设计说明

- 不把 extension 结果放进 `labels`，避免污染 core 消费方。
- extension 置信度与 unmapped 独立于 core `confidence` / `unmapped`。
- `matched` 与 `status` 都保留，方便统计 trigger 命中率与执行结果。

---

## Inline 持久化设计

当前 inline 结构位于：

- `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_pass1.py`
- `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_labels.py`
- `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/inline_scoring.py`

v1 方案：

### turn 记录新增字段

```json
{
  "turn_index": 1,
  "sample_id": "...",
  "labels": { ...core labels... },
  "label_extensions": {
    "ui_fine_labels": { ... }
  },
  "labeling_monitor": { ... }
}
```

其中每个 `label_extensions.<id>` 至少包含：

```json
{
  "status": "success",
  "matched": true,
  "spec_version": "v1",
  "spec_hash": "sha256:...",
  "labels": { ... },
  "confidence": { ... },
  "unmapped": [],
  "monitor": { ... }
}
```

这意味着以下 inline 合同都必须同步扩展，否则数据会在重建链路中丢失：

- `inline_pass1.py::_turn_record()` 生成 turn record 时写入 `label_extensions`
- `inline_labels.py::compact_turn_record()` 保留 `label_extensions`
- `inline_scoring.py::_labels_from_turn_record()` 之外增加对 `label_extensions` 的恢复
- inline rebuilt cache (`labeled.json` / `labeled.jsonl`) 也要把 `label_extensions` 带回样本级 payload

### meta 新增字段

在 `data_label.meta` 中增加 extension 运行指纹，例如：

```json
{
  "extension_specs": {
    "ui_fine_labels": {
      "spec_version": "v1",
      "spec_hash": "sha256:..."
    }
  }
}
```

### 关键兼容规则

1. 旧 inline row 没有 `label_extensions` 时必须可正常读取。
2. compaction 不能丢掉 `label_extensions`。
3. mirrored cache rebuild 必须能从 inline row 重新恢复 extension 数据。
4. 仅 extension 变化时，默认 **不清空 Pass2**，因为 v1 extension 不参与 scoring。
5. inline planner 必须显式区分：
   - core label changed
   - extension label changed
   - extension spec fingerprint changed

也就是说，当前只有 `pass1_changed/invalidate_pass2` 两个粗粒度开关还不够；实现时需要至少补出“core changed vs extension-only changed”的判断能力。

---

## Stats 设计

为避免污染现有 Pass1 stats 结构，v1 推荐采用平行扩展字段：

```json
{
  "total_samples": 1000,
  "tag_distributions": { ...core... },
  "extension_stats": {
    "ui_fine_labels": {
      "display_name": "UI Fine Labels",
      "matched": 180,
      "success": 172,
      "failed": 8,
      "fields": {
        "component_type": {
          "type": "multi_enum",
          "distributions": {
            "form": 80,
            "table": 57
          },
          "confidence": {
            "mean": 0.81,
            "min": 0.43,
            "max": 0.96,
            "count": 172
          }
        },
        "visual_complexity": {
          "type": "enum",
          "distributions": {
            "low": 20,
            "medium": 98,
            "high": 54
          }
        }
      }
    }
  }
}
```

### 统计边界

- 现有 `tag_distributions` / `combo_distributions` / `cross_matrix` 保持 core-only。
- extension 自己维护 `matched/success/failed` 与 field-level distributions。
- conversation 聚合是否需要 extension 级视图，v1 可以先采用“按样本为主，conversation 只做简单 union/last-win”。

---

## Dashboard 设计

## 总体策略

不把 extension 结果混进现有 core taxonomy 图表，而是在 Pass1 dashboard 中增加独立区块。

### v1 展示结构

```text
Pass1 Dashboard
  - Labeling Overview (core)
  - Unmapped Tags (core)
  - Tag Distributions (core)
  - Extension Labels
      - UI Fine Labels
      - Mobile Fine Labels
      - ...
  - Confidence (core)
  - Intent × Difficulty (core)
  - Coverage (core)
```

### 每个 extension 展示内容

- `matched` / `success` / `failed` / `skipped`
- 每个字段的分布表
- 每个字段的 confidence summary
- 基于现有 explorer 的基础联动

### why

- extension 没有 core taxonomy 的 pool coverage 语义；
- extension 也不一定存在统一 cross-matrix；
- 独立区块更易理解，也更利于后续多 extension 并存。

### dashboard 载荷位置

建议在 Pass1 scope detail payload 中增加：

```json
pass1.modes.sample.extension_sections
pass1.modes.conversation.extension_sections
```

而不是把 extension 混进现有 `distributions`。

同时需要定义清楚以下前端合同：

- `dashboard_aggregation.py`：负责把 `extension_stats` 转成 `extension_sections`
- `dashboard_scopes.py`：负责 scope 级聚合结果里保留 `extension_sections` 所需原始统计
- `dashboard_explorer.py`：需要把 extension tags 平铺成独立 namespace（例如 `ext:ui_fine_labels:component_type=form`），避免与 core label chips 混淆
- `dashboard.js`：需要按 extension -> field 的层级渲染，不依赖 core coverage/cross-matrix 语义

---

## CLI 与交互式启动设计

## CLI

在 `run` 命令上新增 repeatable 参数：

```bash
uv run sft-label run \
  --input data.json \
  --score \
  --label-extension /path/to/ui.yaml \
  --label-extension /path/to/mobile.yaml
```

建议参数：

- `--label-extension <path>`：可重复指定多个 extension spec

可选后续参数（非 v1 必需）：

- `--label-extension-dir <dir>`
- `--label-extension-disable <id>`

## Interactive `start`

在 `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/launcher.py` 中增加：

1. 是否启用 extension labeling
2. 输入一个或多个 extension spec 路径
3. 校验配置并展示摘要
4. dry-run 时把 extension 参数体现在命令中

目标是让用户无需手写复杂 CLI 也能启用这项能力。

---

## 错误处理与降级策略

### extension 失败时

- core 成功保留
- extension 记录 `status=failed`
- 样本整体不转为 partial core labels
- dashboard / stats 中显示失败数

### pipeline 返回合同

`label_one()` 不能只“模糊地返回 labels + label_extensions + monitor”，而应形成稳定合同。例如：

```python
(sample_idx, core_labels, extension_payloads, monitor)
```

其中：

- `core_labels`：现有 Pass1 core labels
- `extension_payloads`：`dict[str, dict]`，即 `label_extensions`
- `monitor`：样本级 monitor，除现有字段外增加 extension 摘要，例如：
  - `extension_calls_total`
  - `extension_statuses`
  - `extension_failures`

这样下游的 `build_sample_artifacts()`、failed-sample accounting、recompute 路径才能稳定消费。

### extension schema 不合法时

- run 前尽早失败（配置校验阶段）
- 给出明确字段和路径错误信息

### stale detection / rerun 规则

标准 run 与 inline run 都需要明确 stale detection：

1. 若某个 extension `spec_hash` 变化：
   - 仅该 extension 的旧结果视为 stale
   - 其他 extension 结果保持有效
   - core labels 默认保持有效
2. 若 extension 被新增：
   - 只对新 extension 做补跑
3. 若 extension 被移除：
   - 旧结果可以保留但标记为 disabled-or-stale，或者在 refresh 模式下清理
4. 若 core labels 变化：
   - 该样本的所有 extension 都应视为 stale 并重算

这套规则必须在标准 run、inline incremental/refresh/migrate/recompute 里保持一致。

### extension 输出不合法时

- 记录 `status=invalid`
- monitor 写出校验失败原因
- 不写入非法标签值

### 老 run / 无 extension

- dashboard 自动隐藏 extension 区域
- recompute / regenerate-dashboard 正常工作
- inline 读取保持兼容

---

## 多 extension 并存规则

1. extension 之间相互独立，不共享字段名空间；以 `id` 为顶级 key。
2. 不同 extension 可以对同一样本同时命中并执行。
3. 统计、dashboard、review/export 均按 extension 分组展示。
4. 若 extension 数量较多，dashboard 按 `dashboard.group` / `priority` 排序。
5. 不允许两个 extension 具有相同 `id`。

---

## 实现边界与模块建议

建议新增独立模块，避免继续把复杂逻辑堆进 `pipeline.py`：

- `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions.py`
  - spec loader
  - spec hash
  - trigger matching
  - response validation
  - prompt building
  - runtime execution entrypoint

- `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions_schema.py`
  - schema-level validation helpers

- `/Users/lxs/.codex/worktrees/e8fb/sft-label/src/sft_label/label_extensions_stats.py`
  - extension stats aggregation

这样可以让：

- `pipeline.py` 只负责 orchestration
- inline 层只负责 persistence
- dashboard 层只消费聚合结果

---

## 测试策略

### 单元测试

- spec 解析与校验
- trigger matcher
- spec hash / version tracking
- extension response validator
- success / skipped / failed / invalid 状态机

### 集成测试

- 单 extension run
- 多 extension 并存 run
- core success + extension fail
- inline persistence / cache rebuild
- regenerate-dashboard / recompute-stats

### 回归测试

- 无 extension 时现有流程完全不变
- Pass2 scoring prompt 不带 extension 数据
- review CSV 默认格式不变
- launcher dry-run 正确生成命令

### dashboard 测试

- extension 区块正确出现
- 无 extension 时自动隐藏
- explorer 至少支持基础查询/展示

---

## 多 agent 开发拆分建议

为了支持后续并行开发，建议从设计上拆成四条主线：

### Workstream A：core runtime

- spec loader / validator
- trigger matcher
- prompt builder
- `pipeline.py` integration

### Workstream B：inline / maintenance

- inline persistence
- compaction
- cache rebuild
- recompute / regenerate compatibility

### Workstream C：dashboard / stats

- extension stats builder
- dashboard payload
- dashboard.js rendering
- explorer support

### Workstream D：CLI / launcher / docs / QA

- CLI 参数
- interactive launcher
- docs / examples
- compatibility tests

每条主线之间通过稳定数据契约协作，而不是共享隐式实现细节。

---

## 决策总结

1. **采用 schema-driven extension labeling，而不是 UI 特判。**
2. **从一开始支持多个 extension 配置并存。**
3. **extension 结果存于 `label_extensions`，与 core `labels` 隔离。**
4. **v1 不影响 Pass2 与核心 rarity/selection。**
5. **dashboard 单独展示 extension 区域。**
6. **inline 记录 extension 结果与 spec 指纹，但默认不触发 Pass2 失效。**
7. **实现时按 runtime / persistence / dashboard / CLI+QA 四条主线并行推进。**

