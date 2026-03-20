---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Inter', 'Segoe UI', 'Noto Sans SC', sans-serif;
    background: #FCFDFE;
    color: #10233F;
    font-size: 23px;
    padding: 36px 56px;
  }
  section.lead {
    background: linear-gradient(135deg, #10233F 0%, #1a3a5c 100%);
    color: #F8FBFF;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.lead h1 { color: #F8FBFF; font-size: 2.4em; }
  section.lead h2 { color: #C8D9FF; font-weight: 400; }
  section.lead p { color: #A0B8D8; }
  section.accent {
    background: linear-gradient(135deg, #EFF4FF 0%, #F4F7FB 100%);
  }
  section.green {
    background: linear-gradient(135deg, #EEF9F4 0%, #F4FBF8 100%);
  }
  section.warm {
    background: linear-gradient(135deg, #FDF8F0 0%, #FFF9F2 100%);
  }
  section.dark {
    background: #10233F;
    color: #F8FBFF;
  }
  section.dark h2 { color: #F8FBFF; }
  section.dark h3 { color: #C8D9FF; }
  section.dark p, section.dark li { color: #DCE8FF; }
  section.dark table th { background: #254670; color: #F8FBFF; }
  section.dark table td { background: #1e3d5f; color: #F8FBFF; border-color: #35537B; }
  section.dark code { background: #254670; color: #C8D9FF; }
  section.dark pre { background: #0a1929 !important; }
  section.dark pre code { color: #DCE8FF; background: transparent; }
  section.dark blockquote { border-left-color: #C8D9FF; color: #A0B8D8; }
  section.dark em { color: #F0C674; }
  h1 { color: #10233F; }
  h2 { color: #315AA8; font-size: 1.2em; }
  h3 { color: #5F6F88; font-size: 0.92em; }
  table { font-size: 0.62em; width: auto; margin: 0 auto; }
  td, th { text-align: center; }
  th { background: #EFF4FF; }
  td, th { padding: 3px 7px; }
  code { background: #EFF4FF; color: #315AA8; font-size: 0.78em; }
  pre { background: #10233F !important; border-radius: 10px; padding: 10px !important; }
  pre code { color: #DCE8FF; background: transparent; font-size: 0.6em; line-height: 1.28; }
  pre code span { color: #DCE8FF !important; }
  pre code .hljs-keyword { color: #7EC8E3 !important; }
  pre code .hljs-string { color: #A8D8A8 !important; }
  pre code .hljs-number { color: #F0C674 !important; }
  pre code .hljs-attr { color: #C8D9FF !important; }
  pre code .hljs-comment { color: #6A7A8A !important; }
  .columns { display: flex; gap: 1.6em; }
  .col { flex: 1; }
  blockquote { border-left: 4px solid #315AA8; padding-left: 1em; color: #5F6F88; font-style: italic; font-size: 0.76em; margin: 4px 0; }
  em { color: #A66A09; font-style: normal; font-weight: 600; }
  ul { margin: 3px 0; }
  li { margin: 1px 0; font-size: 0.88em; }
  p { margin: 4px 0; }
---

<!-- _class: lead -->

# Extension Labeling

## 可插拔的子领域标注扩展

在核心 9 维分类之上，按需挂载领域专属标签

---

<!-- _class: accent -->

## 它是什么 & 在流水线中的位置

<div class="columns">
<div class="col">

Extension Labeling 让你用一个 *YAML 配置文件* 定义子领域标签，挂载到 Pass 1 之后运行。

```
Pass 1 Call 1 → 5 基础维度
Pass 1 Call 2 → 4 上下文维度
    ↓ 合并 & 一致性校验
    ↓
  ✨ Extension Labeling ← 在这里
    ↓
Pass 2 → 价值评分（不受扩展影响）
```

- 扩展标签存储在独立命名空间 `label_extensions.<spec_id>`
- 默认不影响核心评分（value_score / selection_score），开启 --extension-rarity-mode bonus_only 后，扩展标签会参与计算独立的 V2 分数列
- 支持同时挂载多个扩展，互相隔离

</div>
<div class="col">

### 典型用途

- 代码子领域的细化标注，比如针对 frontend 数据，自定义新的标签集合，补充标注（Dashboard 默认**提供**可视化）
- 甚至可用于，自定义插入任何 LLM-as-Judge 的标注行为（Dashboard 默认**不提供**可视化）

### 示例文件在哪

| 文件 | 定位 |
|------|------|
| `extensions/ui_web_analysis_v1.example.yaml` | 推荐，Launcher 默认加载，可直接试用 |
| `docs/examples/extensions/ui_fine_labels_minimal_v1.yaml` | 已归档，最小化入门 |
| `docs/examples/extensions/ui_fine_labels_v1.yaml` | 已归档，扩展标注 |
| `docs/examples/extensions/ui_web_analysis_v1.yaml` | 已归档，配比分析 |

</div>
</div>

---

## 案例讲解 — 配置文件的结构（上半部分）

以 `extensions/ui_web_analysis_v1.example.yaml` 为例：

<div class="columns">
<div class="col">

### 基础信息

```yaml
id: ui_web_analysis_example       # 唯一 ID
spec_version: v1                   # 版本号
display_name: Web UI Analysis Labels (Example)
enabled: true
description: Example extension for auditing
  Web UI SFT dataset mix.
```

- `id`：全局唯一，决定输出中的 key 名
- `spec_version`：版本追踪，输出会带上此值
- `enabled`：设为 `false` 可禁用

</div>
<div class="col">

### trigger 规则过滤

```yaml
trigger:
  domain_any_of: [web-frontend]   # 宽松路由
  language_any_of: []             # 空 = 不过滤, [javascript, typescript]
  intent_any_of: []               # ex: [build, modify]
  task_any_of: []
  context_any_of: []
  difficulty_any_of: []
```

- trigger 基于核心 Pass 1 标签（**注意：并非 100% 准确**）做路由，决定哪些样本触发该扩展
- 如果只标注小规模特定数据文件，可以完全不填写过滤规则，默认全标

| 规则 | 含义 |
|------|------|
| `*_any_of: [a, b]` | OR — 命中任一即通过 |
| `*_all_of: [a, b]` | AND — 必须全部包含 |
| `*_any_of: []` | 该维度不参与过滤 |
| 不同维度之间 | *AND* 关系 |


</div>
</div>

---

## 案例讲解 — 配置文件的结构（下半部分）

<div class="columns">
<div class="col">

### prompt — 告诉 LLM 标注什么

```yaml
prompt: |
  你是 Web UI SFT 数据分析标注器。
  这些标签只用于数据覆盖/配比分析，
  不用于评价回答质量。
  只分析用户可见的 Web / desktop browser UI；
  不确定字段留空。
  严格按 schema 输出 JSON，不要输出解释。
```

写 prompt 的要点：
- 明确标注目标和判断标准，告诉 LLM "不确定字段留空"，要求只输出 JSON，不要解释，prompt 总长度整体控制在 *≤ 2,000 字符*（含 schema）

### output — 输出控制

```yaml
output:
  include_confidence: true  # 每字段置信度 0-1
  allow_unmapped: true       # 记录超出选项的值
dashboard:
  group: ui                  # Dashboard 分组
  priority: 12               # 排序权重
```

</div>
<div class="col">

### schema — 定义标签字段

```yaml
schema:
  ui_surface_type:                    # 字段名
    type: enum                        # 单选
    options:                          # 选项列表
      - marketing-or-content-page
      - app-shell-or-settings
      - ...
    description: 按页面主要用途分类     # 判断标准

  ui_constraint_focus:
    type: multi_enum                  # 多选
    options:
      - design-system-consistency
      - responsive-layout
      - ...
    description: 样本涉及的 UI 工程约束
```

两种字段类型：
- `enum`：单选，LLM 返回字符串
- `multi_enum`：多选，LLM 返回字符串数组

建议每个扩展 *3-5 个字段*，每字段 *≤ 20 选项*。

</div>
</div>

---

<!-- _class: green -->

## 照着写您自己的扩展

<div class="columns">
<div class="col">

### 四步完成

1. 复制 `extensions/ui_web_analysis_v1.example.yaml`
2. 改 `id`、`display_name`、`prompt`、`schema`
3. 设置 `trigger`（先只用最宽维度）
4. 保存到 `extensions/` 目录

### 设计要点

| 要点 | 建议 |
|------|------|
| 字段数 | 3-5 个 |
| 选项数 | 每字段 ≤ 20 |
| prompt + schema | ≤ 2,000 字符 |
| 字段名 | `snake_case` |
| 选项名 | `kebab-case` |
| 不同领域 | 拆成独立 spec |

</div>
<div class="col">

### 示例：移动端数据标注扩展

```yaml
id: mobile_interaction
spec_version: v1
display_name: Mobile Interaction Labels
enabled: true
trigger:
  domain_any_of: [mobile-ios, mobile-android]
prompt: |
  你是移动端 UI 标注器。
  只分析移动端原生或跨平台 UI。
  严格按 schema 输出 JSON。
schema:
  platform:
    type: enum
    options: [ios-native, android-native,
      react-native, flutter, other]
    description: 目标平台
  gesture_pattern:
    type: multi_enum
    options: [tap, swipe, long-press,
      pull-to-refresh, pinch-zoom]
    description: 涉及的手势交互
  navigation_type:
    type: enum
    options: [tab-bar, drawer, stack, modal]
    description: 主导航模式
output:
  include_confidence: true
  allow_unmapped: true
```

写好后放到 `extensions/` 目录即可。

</div>
</div>

---

<!-- _class: warm -->

## 启动标注

<div class="columns">
<div class="col">

### 交互式启动（推荐）

```bash
uv run sft-label start
```

Launcher 自动扫描 `extensions/` 目录，列出可用扩展供你多选。启用后会提示成本：每个扩展对每个 matched 样本增加一次 LLM 调用。

### 命令行启动

```bash
uv run sft-label run \
  --input data.jsonl \
  --label-extension extensions/my_ext.yaml

# 多个扩展（flag 可重复）
uv run sft-label run \
  --input data.jsonl \
  --label-extension extensions/web.yaml \
  --label-extension extensions/mobile.yaml
```

> 建议先加 `--limit 20` 小批量验证，确认 matched/skipped 比例合理后再全量运行。

</div>
<div class="col">

### 查看结果

扩展统计自动出现在 Dashboard 中，或者查看数据文件的新增标注字段，导出 CSV 时加 `--include-extensions` 可包含扩展列：

```bash
uv run sft-label export-review --input run_dir --output review.csv \
  --include-extensions
```

### 输出结构（每个样本）

```json
{
  "label_extensions": {
    "ui_web_analysis_example": {
      "status": "success",
      "matched": true,
      "labels": {
        "ui_surface_type": "data-dense-dashboard",
        "ui_constraint_focus": [
          "responsive-layout"]},
      "confidence": {
        "ui_surface_type": 0.92},
      "unmapped": []
    }}}
```

</div>
</div>

---

<!-- _class: dark -->

## 速查

| 主题 | 要点 |
|------|------|
| 示例文件 | `extensions/ui_web_analysis_v1.example.yaml`（推荐起步） |
| 参考示例 | `docs/examples/extensions/*.yaml`（3 个变体） |
| 完整参考指南 | `docs/guides/pass1-extension-labeling.md` |
| 成本估算 | 额外 LLM 调用 ≈ matched 样本数 × 启用扩展数 |
| Trigger 策略 | 分析型用宽松（仅 domain）；精确标注收紧多维度 |
| 何时拆分 | 不同领域、schema 超 5 字段、prompt 超 2K 字符 |
| Inline 存储 | `extra_info.unique_info.data_label.turns[].label_extensions` |
| Extension Rarity V2 | `--extension-rarity-mode {off,preview,bonus_only}`（可选） |
| 源码 | `src/sft_label/label_extensions*.py`（4 个文件） |
