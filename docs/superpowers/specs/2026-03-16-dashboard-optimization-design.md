# Dashboard 目录化与共享静态资源优化设计

## 背景

当前 dashboard 通过 `render_dashboard_html()` 将完整 payload 直接内嵌到 `dashboard_*.html`。在目录级大规模运行（例如 41 个文件、约 15 万条样本）时，单个 HTML 可膨胀到 272 MB，导致：

- 浏览器首次打开极慢，甚至卡死
- 同一台服务器上保存多个 run 时，重复存储大量相同 JS/CSS/runtime
- 通过 URL 分享时，单页传输成本过高

现有 explorer 已经把部分样本级数据拆到 `.assets/` sidecar JS 分片，但 dashboard 主体仍然把大量聚合结果一次性塞进 HTML，且存在明显冗余（例如 `conf_matrix` 与 sample-mode payload 重复）。

## 目标

1. 保持“每个 run 一个独立 URL”的分享方式。
2. 将 dashboard HTML 缩小为轻量 bootstrap 页面。
3. 将通用前端 runtime 改为共享静态资源，适配服务器上大量 run 共存。
4. 将 run 绑定的数据拆成按需加载的 manifest / scope detail / explorer 分片。
5. 保持现有 scope tree、聚合视图、explorer drill-down 功能可用。
6. 为 e2e 与回归测试提供稳定的产物结构和体积约束。

## 非目标

- 不引入后端 API 依赖；dashboard 仍可由静态文件托管。
- 不追求“单文件离线 HTML”继续承载超大数据集。
- 不在本次改造中重做 dashboard 的视觉设计。

## 方案概览

将 dashboard 产物拆分为三层：

1. **轻量 HTML**：每个 run 仍然输出 `dashboard_labeling.html` / `dashboard_scoring.html`，但只包含 bootstrap 配置与共享静态资源引用。
2. **共享静态资源**：前端 JS/CSS/runtime、i18n 文案、常量表统一部署到固定路径，例如 `/static/sft-label-dashboard/v1/`。
3. **run 私有数据目录**：每个 dashboard 对应一个 `.data/` 目录，保存 manifest、scope detail、explorer 分片等运行数据。

示例结构：

```text
/static/sft-label-dashboard/v1/
  dashboard.js
  dashboard.css

/runs/<run-id>/dashboards/
  dashboard_labeling.html
  dashboard_labeling.data/
    manifest.json
    scopes/
      global.json
      dir__code.json
      file__code__magicoder_oss_instruct.json.json
    explorer/
      preview_*.js
      detail_*.js
  dashboard_scoring.html
  dashboard_scoring.data/
    manifest.json
    scopes/
      ...
    explorer/
      ...
```

## 前端加载模型

### HTML bootstrap

HTML 不再内嵌 `const DATA = {...}`。改为仅输出：

- dashboard 类型（labeling / scoring）
- run 私有数据目录相对路径（例如 `dashboard_labeling.data/`）
- 静态资源版本 / 可选绝对 URL 前缀
- 默认语言、默认 scope 等少量初始化参数

页面加载顺序：

1. HTML 加载共享 `dashboard.css` 与 `dashboard.js`
2. runtime 拉取 `manifest.json`
3. 首屏渲染 scope tree 与 summary
4. 用户切换 scope 时按需加载 `scopes/<scope-id>.json`
5. 用户打开 explorer 时按需加载 preview/detail 分片

### 共享静态资源

共享资源包含：

- dashboard 页面主 runtime JS
- 样式 CSS
- i18n 文案与常量（例如维度名、颜色、排序选项）
- 通用 loader、状态恢复、scope/render/explorer 逻辑

这些资源不再随每个 run 重复写入；浏览器与 CDN 可复用缓存。

## 数据拆分设计

### 1. manifest.json（首屏必需）

manifest 仅保留轻量信息：

- `title` / `subtitle`
- `root_id` / `default_scope_id`
- `initially_expanded`
- 轻量 scope tree：`id`、`label`、`kind`、`path`、`parent_id`、`children`
- 每个 scope 的 summary：
  - `file_count`
  - `pass1_total`
  - `scored_total`
  - `mean_value`
  - 是否存在 `pass1` / `pass2` / `conversation` / `explorer`
- explorer 顶层开关和轻量配置

manifest 不携带完整 pass1/pass2 聚合数据，不包含样本级内容。

### 2. scope detail JSON（按 scope 懒加载）

每个 scope 一个 detail JSON，保存该 scope 的完整可视化数据。首屏只请求默认 scope；切换时再拉取其他 scope。

#### Pass 1 保留字段

- `modes.sample` / `modes.conversation`
- `distributions`
- `confidence_stats`
- `coverage`
- `cross_matrix`
- `unmapped_details`（保留 top N）
- `overview`

#### Pass 1 删除/缩减

- **删除 `conf_matrix`**：当前体积大且模板未直接消费，默认不导出到 dashboard 数据。
- **移除顶层 sample-mode 重复结构**：只保留 `modes`，避免顶层与 `modes.sample` 双份存储。
- `unmapped_details.examples` 限流：每 tag 默认 0~1 条示例。
- 所有浮点数统一 round，降低 JSON 体积。

#### Pass 2 保留字段

- `modes.sample` / `modes.conversation`
- `overview`
- `histograms`
- `score_distributions`
- `value_by_tag`
- `selection_by_tag`
- `thinking_mode_stats`
- `flag_counts`
- `flag_value_impact`
- `coverage_at_thresholds`
- `weights_used`
- `per_file_summary`
- `conversation`

#### Pass 2 缩减策略

- `value_by_tag` / `selection_by_tag` 支持裁剪长尾（例如仅保留 top/bottom N，或按最小样本数过滤）。
- 目录级 scope 的 `per_file_summary` 避免在祖先 scope 中重复展开过大列表；必要时仅保留直接子节点或 top files。
- 数值统一 round，避免高精度长串。

### 3. explorer 分片（run 私有）

explorer 继续保留 sidecar 资源，但改为 dashboard 私有数据目录的一部分，例如 `dashboard_labeling.data/explorer/`。

保留策略：

- `preview` 分片只保留列表检索所需字段
- `detail` 分片只在用户点开样本时加载
- conversation drill-down 仍随 detail 或单独挂载
- manifest / scope detail 不再重复包含 explorer 摘要数据

## Python 导出层改造

### dashboard_template.py

当前 `render_dashboard_html(payload)` 直接把大 payload 内嵌进模板。改造后：

- 抽出共享 runtime（JS/CSS）生成逻辑
- HTML 模板只渲染 bootstrap 配置
- 前端通过 fetch 读取 manifest 与 scope detail

### visualize_labels.py / visualize_value.py

改造 dashboard 生成流程：

- 输出轻量 HTML
- 输出对应 `.data/manifest.json`
- 输出 `.data/scopes/*.json`
- 输出 `.data/explorer/*`
- 根据配置写入共享静态资源引用，而非复制所有 runtime 到 run 目录

### dashboard_aggregation.py

增加 dashboard 导出专用瘦身层，区分：

- 内部完整聚合结果
- manifest summary
- scope detail payload

避免把内部中间结构原样全部导出到前端。特别是：

- `build_pass1_viz()` / `build_pass2_viz()` 产物改成仅保留 `modes`
- `conf_matrix` 默认不参与 dashboard 导出
- 为 manifest 提供轻量 summary builder
- 为 scope detail builder 统一做裁剪/round/filter

## URL 与部署约定

- 每个 run 仍通过独立 URL 访问：
  - `.../dashboards/dashboard_labeling.html`
  - `.../dashboards/dashboard_scoring.html`
- HTML 通过相对路径寻找 `.data/` 目录，便于迁移整个 run 目录。
- 共享静态资源采用版本路径，例如 `/static/sft-label-dashboard/v1/`。
- HTML bootstrap 中记录静态资源版本，确保旧 run 可继续绑定旧 runtime。

## 向后兼容

1. **文件命名兼容**：继续遵守现有 `PASS1_DASHBOARD_FILE` / `PASS2_DASHBOARD_FILE` 命名。
2. **目录布局兼容**：`dashboards/` 仍为运行期 dashboard 根目录；新增 `.data/` 作为同名 sidecar 目录。
3. **老产物读取兼容**：现有 layout optimizer / regenerate-dashboard / recompute 逻辑继续识别旧 HTML 文件名；新逻辑只改变 dashboard 内容结构。
4. **静态资源可配置**：支持默认相对路径或显式静态前缀，方便不同服务器部署。

## 测试与验证

### 单元/集成测试

- `dashboard_template`：验证 HTML 不再嵌入巨型 payload，而是包含 bootstrap 与共享静态资源引用。
- `visualize_labels` / `visualize_value`：验证会生成 HTML + `.data/manifest.json` + `scopes/` + `explorer/`。
- `dashboard_aggregation`：验证 `conf_matrix` 不导出；`modes` 不重复；summary/detail builder 输出正确。
- explorer：验证 preview/detail 仍可被前端定位和加载。

### e2e（目录输入）

基于 `tests/fixtures/e2e_folder_test/`：

- 目录模式生成 labeling/scoring dashboard
- 生成的 HTML 位于 `dashboards/`
- 对应 `.data/` 目录存在且结构完整
- HTML 使用共享静态资源路径
- scope drill-down 与 explorer 元数据存在
- 多文件目录输入下，每个 file scope 仍可定位到数据源

### 体积回归

新增回归断言：

- HTML 本体显著小于旧实现（例如不允许重新内嵌全量 payload）
- manifest 体积受限
- 单个 scope detail 不包含 `conf_matrix`
- explorer detail 仅在 sidecar 目录出现，不回流进 HTML 或 manifest

## 风险与缓解

### 风险 1：静态资源路径配置复杂

缓解：为 runtime 提供统一 `static_base_url` 配置；默认相对路径，允许部署时覆盖。

### 风险 2：前端从同步内嵌数据切换为异步加载后状态管理更复杂

缓解：显式引入 manifest cache / scope detail cache，并保持现有 hash/state 恢复语义不变。

### 风险 3：scope detail 数量多时请求数增加

缓解：仅按需加载当前 scope；用户通常不会展开所有 scope。必要时可增加内存 cache。

### 风险 4：祖先 scope 聚合 detail 仍可能很大

缓解：通过导出层裁剪冗余字段，并对长尾排名/明细做 limit；优先保留 summary，再保留必要 drill-down。

## 分阶段实施建议

1. 抽离共享 JS/CSS/runtime，改 HTML 为 bootstrap 壳。
2. 引入 `manifest + scopes/*.json` 数据布局，并接入前端异步加载。
3. 在导出层删除 `conf_matrix`、去掉重复 sample payload，并增加裁剪策略。
4. 调整 explorer 目录归属与引用路径。
5. 补齐单元测试、e2e、体积回归测试。

## 预期结果

完成后，dashboard 将从“单个超大 HTML”转为“轻量 HTML + run 私有数据目录 + 共享静态资源”。这能显著降低单页体积与首屏卡顿，支持服务器上大量 run 长期并存，并保持“每个 run 一个独立 URL”的分享方式。
