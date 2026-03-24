(async function () {
window.__SFT_DASHBOARD_DATA__ = window.__SFT_DASHBOARD_DATA__ || { manifest: null, scopes: {} };
const BOOTSTRAP = (() => {
  const node = document.getElementById("dashboard-bootstrap");
  if (!node) return {};
  try {
    return JSON.parse(node.textContent || "{}");
  } catch (error) {
    console.error("Failed to parse dashboard bootstrap", error);
    return {};
  }
})();

const EXPLORER_PREVIEW_CACHE_LIMIT = 6;
const EXPLORER_DETAIL_CACHE_LIMIT = 18;
const EXPLORER_PROGRESS_RENDER_MS = 120;
const EXPLORER_DRAWER_TURN_LIMIT = 12;
const EXPLORER_DRAWER_TEXT_LIMIT = 1600;
const EXPLORER_DRAWER_JSON_LIMIT = 24000;
const EXPLORER_CACHE = {
  scripts: new Map(),
  previews: new Map(),
  details: new Map(),
  previewAssets: new Map(),
  detailAssets: new Map(),
  lastProgressRender: 0,
};
const SCOPE_DETAIL_CACHE = new Map();

function resolveDashboardUrl(path) {
  return new URL(String(path || ""), window.location.href).toString();
}

async function fetchDashboardJson(path) {
  const response = await fetch(resolveDashboardUrl(path), { credentials: "same-origin" });
  if (!response.ok) {
    throw new Error(`Failed to load dashboard payload: ${path} (${response.status})`);
  }
  return await response.json();
}

function dashboardDataRegistry() {
  return window.__SFT_DASHBOARD_DATA__ || { manifest: null, scopes: {} };
}

async function ensureDashboardScriptLoaded(path) {
  if (!path) return;
  return await ensureExplorerScriptLoaded(resolveDashboardUrl(path));
}

async function loadDashboardManifest() {
  if (BOOTSTRAP.manifestScriptUrl) {
    await ensureDashboardScriptLoaded(BOOTSTRAP.manifestScriptUrl);
    const manifest = dashboardDataRegistry().manifest;
    if (manifest) {
      if (!manifest.root_id && BOOTSTRAP.rootId) manifest.root_id = BOOTSTRAP.rootId;
      if (!manifest.default_scope_id && BOOTSTRAP.defaultScopeId) manifest.default_scope_id = BOOTSTRAP.defaultScopeId;
      if (!Array.isArray(manifest.initially_expanded)) {
        manifest.initially_expanded = [manifest.root_id].filter(Boolean);
      }
      return manifest;
    }
  }
  const manifest = await fetchDashboardJson(BOOTSTRAP.manifestUrl);
  if (!manifest.root_id && BOOTSTRAP.rootId) manifest.root_id = BOOTSTRAP.rootId;
  if (!manifest.default_scope_id && BOOTSTRAP.defaultScopeId) manifest.default_scope_id = BOOTSTRAP.defaultScopeId;
  if (!Array.isArray(manifest.initially_expanded)) {
    manifest.initially_expanded = [manifest.root_id].filter(Boolean);
  }
  return manifest;
}

const DATA = await loadDashboardManifest();
const STATE = {
  currentId: DATA.default_scope_id || DATA.root_id,
  locale: "en",
  filter: "all",
  query: "",
  expanded: new Set(DATA.initially_expanded || [DATA.root_id]),
  sidebarCollapsed: false,
  aggregationMode: "sample",
  tagBarMode: "relative",
  keepRateThreshold: "7.0",
  fileRankingSort: { key: "mean_value", direction: "desc" },
  explorer: {
    runId: 0,
    busy: false,
    scanned: 0,
    matched: 0,
    scopeTotal: 0,
    scopeDone: 0,
    status: "",
    results: [],
    sort: (DATA.explorer || {}).result_limit ? "quality_asc" : "sample_id_asc",
    limit: (DATA.explorer || {}).result_limit || 200,
    detailDocId: "",
    detailScopeId: "",
    detailData: null,
    detailLoading: false,
    queryModel: {
      tagQuery: "",
      language: "",
      intent: "",
      sourcePath: "",
      textQuery: "",
      flagQuery: "",
      convValueMin: "",
      convSelectionMin: "",
      convSelectionMax: "",
      peakComplexityMin: "",
      turnCountMin: "",
      observedTurnRatioMin: "",
      observedTurnRatioMax: "",
      rarityConfidenceMin: "",
      rarityConfidenceMax: "",
      minValue: "",
      maxValue: "",
      maxQuality: "",
      minSelection: "",
      maxSelection: "",
      maxConfidence: "",
      thinkingMode: "",
      hasFlags: false,
      includeInherited: false,
    },
  },
};

const I18N = {
  en: {
    dashboard_title_labeling: "SFT Labeling Dashboard",
    dashboard_title_scoring: "SFT Labeling & Scoring Dashboard",
    dashboard_title_generic: "Interactive Dashboard",
    scope_navigator: "Scope Navigator",
    scope_navigator_sub: "Global, folders, and files",
    collapse_navigator: "Collapse Navigator",
    expand_navigator: "Expand Navigator",
    search_folders_files: "Search folders or files",
    filter_all: "All",
    filter_folders: "Folders",
    filter_files: "Files",
    up_one_level: "Up One Level",
    back_to_global: "Back To Global",
    global_overview: "Global overview",
    scopes: "Scopes",
    files: "Files",
    labeled: "Labeled",
    scored: "Scored",
    mean_value: "Mean Value",
    no_data: "No data",
    no_dashboard_data: "No dashboard data for this scope.",
    no_example_sample: "No example sample loaded for this scope.",
    no_conversation_turns: "No conversation turns.",
    no_tags: "No tags",
    loading_sample: "Loading sample…",
    loading_full_sample_payload: "Loading full sample payload…",
    children: "Children",
    name: "Name",
    type: "Type",
    direct_children: "{count} direct children",
    items: "items",
    samples: "samples",
    slices: "slices",
    units: "units",
    aggregation: "Aggregation",
    language: "Language",
    languages: "Languages",
    chinese: "中文",
    english: "English",
    language_toggle_help: "Switch dashboard language",
    aggregation_help_title: "Aggregation mode",
    aggregation_help_body: "Sample: every multi-turn slice contributes separately.\nConversation: multi-turn samples are merged into one unit; single-select tags keep the final labeled state, while multi-select tags mean the tag appeared at least once in the conversation.\n\nTokens, arbitration, and inherited slice counts remain process-level metrics and do not change with this toggle. Conversation Aggregation stays conversation-level in both modes.",
    agg_sample: "Sample",
    agg_conversation: "Conversation",
    agg_sample_hint: "Count each multi-turn slice separately",
    agg_conversation_hint: "Count each multi-turn conversation once after merging labels",
    tag_bars: "Tag Bars",
    hide: "Hide",
    normalized: "Relative",
    global_log: "Global Log",
    hide_tag_bars: "Hide tag bars",
    relative_hint: "Scale by current panel max",
    global_log_hint: "Scale by section-wide log max",
    tag_bar_help: "Relative scales by the current panel max; Global Log scales by the log of tag counts in the section.",
    labeling_overview: "Labeling Overview",
    units_label: "Units",
    success: "Success",
    tokens: "Tokens",
    arbitrated: "Arbitrated",
    unmapped: "Unmapped",
    llm_labeled: "LLM Labeled",
    inherited: "Inherited",
    llm_labeled_units: "With LLM labels",
    inherited_units: "With inherited turns",
    pass1_conversation_note: "Conversation mode merges one conversation into one unit. Single-select dimensions keep the final labeled state; multi-select dimensions mean the tag appeared at least once. 'With LLM labels' / 'With inherited turns' indicate whether a conversation contains at least one such turn.",
    pass1_scoring_subset_note: "On scoring dashboards, this Pass 1 view follows the conversations that reached scoring in the current scope, not the full Pass 1 census.",
    pass2_conversation_note: "Conversation-mode scoring aggregates one scored conversation per unit. Label breakdowns here reflect scored conversations, not the full Pass 1 census.",
    conversation_panel_note: "Conversation Aggregation is always conversation-level. conv_value / conv_selection are ranking signals for filtering and prioritization, not raw expert quality labels.",
    unmapped_tags: "Unmapped Tags",
    top_10_per_dimension: "{count} occurrences · top 10 per dimension",
    tag: "Tag",
    count: "Count",
    turn_kind_single: "1T",
    turn_kind_multi: "MT",
    turn_kind_mixed: "Mix",
    examples: "Examples",
    tag_distributions: "Tag Distributions",
    all_tags_sorted_frequency: "All tags, sorted by frequency",
    confidence: "Confidence",
    dimension: "Dimension",
    mean: "Mean",
    min: "Min",
    max: "Max",
    below_threshold: "Below Threshold",
    intent_difficulty: "Intent × Difficulty",
    pool_coverage: "Pool Coverage",
    unused_tags: "{count} unused tags",
    scoring_overview: "Scoring Overview",
    failed: "Failed",
    complexity: "Complexity",
    quality: "Quality",
    median_rarity: "Median Rarity",
    score_distributions: "Score Distributions",
    confidence_distribution: "Confidence Distribution",
    llm_confidence: "LLM Confidence",
    value_by_tag: "Value By Tag",
    selection_by_tag: "Selection By Tag",
    all_tags_sorted_score: "All tags, sorted by score",
    analysis: "Analysis",
    thinking_mode: "Thinking Mode",
    mode: "Mode",
    value: "Value",
    value_score_v2: "Value V2",
    selection: "Selection",
    selection_v2: "Selection V2",
    reasoning: "Reasoning",
    extension_rarity_preview: "Extension Rarity (Preview)",
    rarity_v2: "Rarity V2",
    extension_rarity_mode: "Extension Rarity Mode",
    extension_baseline_source: "Extension Rarity Baseline",
    flags: "Flags",
    file_ranking: "File Ranking",
    sorted_by: "{count} files · sorted by {column}",
    mean_rarity: "Mean Rarity",
    keep_rate_7: "Keep ≥7",
    keep_rate_threshold: "Keep Rate",
    configuration: "Configuration",
    selection_thresholds: "Selection Thresholds",
    band: "Band",
    threshold: "Threshold",
    action: "Action",
    inspect_samples: "Inspect Slices",
    coverage_at_thresholds: "Coverage At Thresholds",
    value_min: "Value Min",
    retained: "Retained",
    sample_pct: "Sample %",
    tag_coverage: "Tag Coverage",
    extension_labels: "Extension Labels",
    extension_summary: "Extension Summary",
    extension_matched: "Matched",
    extension_status: "Status",
    extension_versions: "Spec Versions",
    extension_hashes: "Spec Hashes",
    extension_fields: "Extension Fields",
    extension_fields_used: "Fields Used",
    extension_values: "Values",
    matched_rate: "Matched Rate",
    extension_prompt: "Prompt",
    extension_schema: "Schema",
    extension_source: "Spec Source",
    extension_field_type: "Field Type",
    extension_option_count: "Options",
    inspect_matched_samples: "Inspect Matched Slices",
    inspect_field_samples: "Inspect Field Slices",
    inspect_extension_value: "Inspect Value Samples",
    extension_configuration: "Extension Configuration",
    extension_details: "Extension Details",
    extension_short_id: "Extension",
    extension_field_short: "Field",
    inspect_retained: "Inspect Retained",
    flag_impact: "Flag Impact",
    value_score: "Value Score",
    conversations: "Conversations",
    conv_value: "Conv Value",
    conv_selection: "Conv Selection",
    peak_complexity: "Peak Complexity",
    mean_turns: "Mean Turns",
    observed_ratio: "Observed Ratio",
    rarity_confidence: "Rarity Confidence",
    convs_lt_50_observed: "{count} convs < 50% observed",
    convs_lt_060_rarity_conf: "{count} convs < 0.60",
    turn_distribution: "Turn Distribution",
    turns_label: "{count} turns",
    observed_turn_coverage: "Observed Turn Coverage",
    conversation_aggregation: "Conversation Aggregation",
    sample_id: "Sample Id",
    quality_asc: "Quality Asc",
    value_asc: "Value Asc",
    confidence_asc: "Confidence Asc",
    selection_desc: "Selection Desc",
    conv_value_desc: "Conv Value Desc",
    conv_selection_desc: "Conv Selection Desc",
    turn_count_desc: "Turn Count Desc",
    observed_ratio_asc: "Observed Ratio Asc",
    observed_ratio_desc: "Observed Ratio Desc",
    rarity_confidence_asc: "Rarity Confidence Asc",
    rarity_confidence_desc: "Rarity Confidence Desc",
    rarity_desc: "Rarity Desc",
    run_query_or_choose_preset: "Run a query or choose a preset to load sample previews.",
    sample: "Sample",
    query: "Query",
    response: "Response",
    tags: "Tags",
    observed: "Observed",
    turns: "Turns",
    source: "Source",
    conv_sel: "Conv Sel",
    conv_rarity: "Conv Rarity",
    thinking: "Thinking",
    peak_cplx: "Peak Cplx",
    conversation: "Conversation",
    json: "JSON",
    conversation_preview: "Conversation Preview ({shown}/{total} turns)",
    preview_limited: "Preview limited to first {limit} turns. {hidden} more turns stay in raw JSON.",
    raw_json_preview: "Raw JSON Preview",
    raw_json_preview_cap: "Raw JSON Preview ({cap} char cap)",
    large_payload_truncated: "Large payload truncated in drawer to keep long multi-turn samples responsive.",
    python_debug: "Python + Debug",
    lowest_quality: "Lowest Quality",
    lowest_value: "Lowest Value",
    low_confidence: "Low Confidence",
    has_flags: "Has Flags",
    long_multiturn: "Long Multi-turn",
    low_coverage: "Low Coverage",
    low_rarity_conf: "Low Rarity Conf",
    sample_explorer: "Slice Explorer",
    explorer_summary: "Progressively scan preview shards, stream large files chunk-by-chunk, keep the best {limit} matches in memory, and click tags / bars / threshold rows anywhere above to drill into matching slices. Explorer counts remain slice-based even when the rest of the dashboard is in conversation mode.",
    status_ready_scan: "Ready to scan {count} indexed slices in this scope.",
    status_scanning: "Scanning {done}/{total} files · {rows} rows checked · {matches} matches",
    status_done: "Done. Matched {matches} slice rows across {files} files.",
    tags_comma: "Tags (comma)",
    source_file: "Source File",
    text_contains: "Text Contains",
    flag: "Flag",
    conv_sel_min: "Conv Sel ≥",
    conv_sel_max: "Conv Sel ≤",
    peak_cplx_min: "Peak Cplx ≥",
    turns_min: "Turns ≥",
    observed_ratio_min: "Observed Ratio ≥",
    observed_ratio_max: "Observed Ratio ≤",
    rarity_conf_min: "Rarity Conf ≥",
    rarity_conf_max: "Rarity Conf ≤",
    value_ge: "Value ≥",
    value_le: "Value ≤",
    quality_le: "Quality ≤",
    selection_ge: "Selection ≥",
    selection_le: "Selection ≤",
    confidence_le: "Confidence ≤",
    any: "Any",
    sort: "Sort",
    only_samples_with_flags: "Only slices with flags",
    include_inherited_labels: "Include inherited labels",
    run_query: "Run Query",
    scanning: "Scanning…",
    reset: "Reset",
    active_filters: "Active Filters",
    clear_all: "Clear All",
    remove_filter: "Remove filter",
    extension_short: "Ext",
    current_scope_summary: "Current scope: {samples} indexed slices · {files} candidate files · showing top {limit} results sorted by {sort}",
    matches_retained: "{matches} matches retained after scanning {rows} rows",
    total: "Total",
    dim_intent: "Intent",
    dim_language: "Language",
    dim_domain: "Domain",
    dim_task: "Task",
    dim_difficulty: "Difficulty",
    dim_concept: "Concept",
    dim_agentic: "Agentic",
    dim_constraint: "Constraint",
    dim_context: "Context",
    scope_kind_global: "Global",
    scope_kind_dir: "Folder",
    scope_kind_file: "File",
    thinking_fast: "Fast",
    thinking_slow: "Slow",
  },
  zh: {
    dashboard_title_labeling: "SFT 标注看板",
    dashboard_title_scoring: "SFT 标注与评分看板",
    dashboard_title_generic: "交互式看板",
    scope_navigator: "范围导航",
    scope_navigator_sub: "全局、目录与文件",
    collapse_navigator: "收起导航",
    expand_navigator: "展开导航",
    search_folders_files: "搜索目录或文件",
    filter_all: "全部",
    filter_folders: "目录",
    filter_files: "文件",
    up_one_level: "返回上一级",
    back_to_global: "返回全局",
    global_overview: "全局视图",
    scopes: "范围",
    files: "文件",
    labeled: "已标注",
    scored: "已评分",
    mean_value: "平均价值",
    no_data: "暂无数据",
    no_dashboard_data: "当前范围没有可展示的看板数据。",
    no_example_sample: "当前范围没有加载示例样本。",
    no_conversation_turns: "没有对话轮次。",
    no_tags: "无标签",
    loading_sample: "正在加载样本…",
    loading_full_sample_payload: "正在加载完整样本内容…",
    children: "子项",
    name: "名称",
    type: "类型",
    direct_children: "共 {count} 个直接子项",
    items: "条",
    samples: "样本",
    slices: "切片",
    units: "单元",
    aggregation: "统计口径",
    language: "语言",
    languages: "语言",
    chinese: "中文",
    english: "English",
    language_toggle_help: "切换看板语言",
    aggregation_help_title: "统计口径",
    aggregation_help_body: "样本：多轮数据按切片分别统计。\n会话：多轮数据先聚合成一条；单选维度保留最终标注状态，多选维度表示该标签在整段会话中至少出现过一次。\n\nTokens、仲裁率、继承切片数等仍然反映处理过程，因此不会随切换而变化。会话聚合面板始终按会话统计。",
    agg_sample: "样本",
    agg_conversation: "会话",
    agg_sample_hint: "多轮按切片分别计数",
    agg_conversation_hint: "多轮先合并后按整段对话计数",
    tag_bars: "标签条",
    hide: "隐藏",
    normalized: "归一化",
    global_log: "全局对数",
    hide_tag_bars: "隐藏标签条",
    relative_hint: "按当前面板最大值缩放",
    global_log_hint: "按当前分区的标签计数做对数缩放",
    tag_bar_help: "归一化按当前面板最大值缩放；全局对数按当前 section 内标签计数的对数缩放。",
    labeling_overview: "标注总览",
    units_label: "单元数",
    success: "成功率",
    tokens: "Tokens",
    arbitrated: "仲裁率",
    unmapped: "未映射",
    llm_labeled: "LLM 标注",
    inherited: "继承",
    llm_labeled_units: "含 LLM 标注",
    inherited_units: "含继承轮次",
    pass1_conversation_note: "会话口径下，一段对话只算一个单元。单选维度表示最终标注状态；多选维度表示该标签在会话中至少出现过一次。“含 LLM 标注” / “含继承轮次”表示该会话是否包含至少一轮此类标注。",
    pass1_scoring_subset_note: "在评分看板中，这个 Pass 1 视图跟随“当前范围内进入评分阶段的会话”，而不是完整的 Pass 1 全量普查。",
    pass2_conversation_note: "会话口径的评分是按“已评分会话”聚合而成。这里的标签分布反映的是 scored conversations，而不是完整的 Pass 1 全量普查。",
    conversation_panel_note: "会话聚合面板始终按会话统计。conv_value / conv_selection 更适合做过滤和优先级排序，不应直接等同于专家主观质量结论。",
    unmapped_tags: "未映射标签",
    top_10_per_dimension: "{count} 次出现 · 每个维度展示前 10 个",
    tag: "标签",
    count: "数量",
    turn_kind_single: "单",
    turn_kind_multi: "多",
    turn_kind_mixed: "混",
    examples: "示例",
    tag_distributions: "标签分布",
    all_tags_sorted_frequency: "所有标签，按频次排序",
    confidence: "置信度",
    dimension: "维度",
    mean: "平均",
    min: "最小",
    max: "最大",
    below_threshold: "低于阈值",
    intent_difficulty: "意图 × 难度",
    pool_coverage: "标签池覆盖率",
    unused_tags: "{count} 个未使用标签",
    scoring_overview: "评分总览",
    failed: "失败",
    complexity: "复杂度",
    quality: "质量",
    median_rarity: "稀有度中位数",
    score_distributions: "分数分布",
    confidence_distribution: "置信度分布",
    llm_confidence: "LLM 置信度",
    value_by_tag: "按标签看价值",
    selection_by_tag: "按标签看入选分",
    all_tags_sorted_score: "所有标签，按分数排序",
    analysis: "分析",
    thinking_mode: "思考模式",
    mode: "模式",
    value: "价值",
    value_score_v2: "价值 V2",
    selection: "入选分",
    selection_v2: "入选分 V2",
    reasoning: "推理",
    extension_rarity_preview: "扩展稀有度（预览）",
    rarity_v2: "稀有度 V2",
    extension_rarity_mode: "扩展稀有度模式",
    extension_baseline_source: "扩展稀有度基线",
    flags: "标记",
    file_ranking: "文件排名",
    sorted_by: "{count} 个文件 · 当前按 {column} 排序",
    mean_rarity: "平均稀有度",
    keep_rate_7: "保留率 ≥7",
    keep_rate_threshold: "保留率",
    configuration: "配置",
    selection_thresholds: "入选分阈值",
    band: "区间",
    threshold: "阈值",
    action: "操作",
    inspect_samples: "查看切片",
    coverage_at_thresholds: "阈值覆盖率",
    value_min: "价值下限",
    retained: "保留数",
    sample_pct: "样本占比",
    tag_coverage: "标签覆盖率",
    extension_labels: "扩展标签",
    extension_summary: "扩展概览",
    extension_matched: "命中",
    extension_status: "状态",
    extension_versions: "版本",
    extension_hashes: "哈希",
    extension_fields: "扩展字段",
    extension_fields_used: "字段数",
    extension_values: "值数",
    matched_rate: "命中率",
    extension_prompt: "Prompt",
    extension_schema: "Schema",
    extension_source: "Spec 来源",
    extension_field_type: "字段类型",
    extension_option_count: "选项数",
    inspect_matched_samples: "查看命中切片",
    inspect_field_samples: "查看字段切片",
    inspect_extension_value: "查看该值样本",
    extension_configuration: "扩展配置",
    extension_details: "扩展详情",
    extension_short_id: "扩展",
    extension_field_short: "字段",
    inspect_retained: "查看保留结果",
    flag_impact: "标记影响",
    value_score: "价值分",
    conversations: "会话数",
    conv_value: "会话价值",
    conv_selection: "会话入选分",
    peak_complexity: "峰值复杂度",
    mean_turns: "平均轮次",
    observed_ratio: "观测占比",
    rarity_confidence: "稀有度置信度",
    convs_lt_50_observed: "{count} 条会话 < 50%",
    convs_lt_060_rarity_conf: "{count} 条会话 < 0.60",
    turn_distribution: "轮次分布",
    turns_label: "{count} 轮",
    observed_turn_coverage: "已观测轮次覆盖",
    conversation_aggregation: "会话聚合",
    sample_id: "样本 ID",
    quality_asc: "质量升序",
    value_asc: "价值升序",
    confidence_asc: "置信度升序",
    selection_desc: "入选分降序",
    conv_value_desc: "会话价值降序",
    conv_selection_desc: "会话入选分降序",
    turn_count_desc: "轮次降序",
    observed_ratio_asc: "观测占比升序",
    observed_ratio_desc: "观测占比降序",
    rarity_confidence_asc: "稀有度置信度升序",
    rarity_confidence_desc: "稀有度置信度降序",
    rarity_desc: "稀有度降序",
    run_query_or_choose_preset: "运行查询或选择预设，以加载样本预览。",
    sample: "样本",
    query: "问题",
    response: "回答",
    tags: "标签",
    observed: "观测",
    turns: "轮次",
    source: "来源",
    conv_sel: "会话入选分",
    conv_rarity: "会话稀有度",
    thinking: "思考",
    peak_cplx: "峰值复杂度",
    conversation: "对话",
    json: "JSON",
    conversation_preview: "对话预览（{shown}/{total} 轮）",
    preview_limited: "预览仅展示前 {limit} 轮，其余 {hidden} 轮可在原始 JSON 中查看。",
    raw_json_preview: "原始 JSON 预览",
    raw_json_preview_cap: "原始 JSON 预览（最多 {cap} 字符）",
    large_payload_truncated: "为保持长多轮样本的响应速度，抽屉中的大内容已截断。",
    python_debug: "Python + 调试",
    lowest_quality: "最低质量",
    lowest_value: "最低价值",
    low_confidence: "低置信度",
    has_flags: "有标记",
    long_multiturn: "长多轮",
    low_coverage: "低覆盖",
    low_rarity_conf: "低稀有度置信度",
    sample_explorer: "切片浏览器",
    explorer_summary: "渐进扫描预览分片，按 chunk 流式读取大文件，仅在内存中保留最匹配的 {limit} 条结果；也可以点击上面的标签、柱状图和阈值行继续向下钻取对应切片。即使上方切换到会话口径，浏览器这里仍按切片索引和筛选。",
    status_ready_scan: "当前范围可扫描 {count} 条已索引切片。",
    status_scanning: "正在扫描 {done}/{total} 个文件 · 已检查 {rows} 行 · 当前匹配 {matches} 条",
    status_done: "完成：在 {files} 个文件中匹配到 {matches} 条切片记录。",
    tags_comma: "标签（逗号分隔）",
    source_file: "来源文件",
    text_contains: "文本包含",
    flag: "标记",
    conv_sel_min: "会话入选分 ≥",
    conv_sel_max: "会话入选分 ≤",
    peak_cplx_min: "峰值复杂度 ≥",
    turns_min: "轮次 ≥",
    observed_ratio_min: "观测占比 ≥",
    observed_ratio_max: "观测占比 ≤",
    rarity_conf_min: "稀有度置信度 ≥",
    rarity_conf_max: "稀有度置信度 ≤",
    value_ge: "价值 ≥",
    value_le: "价值 ≤",
    quality_le: "质量 ≤",
    selection_ge: "入选分 ≥",
    selection_le: "入选分 ≤",
    confidence_le: "置信度 ≤",
    any: "任意",
    sort: "排序",
    only_samples_with_flags: "只看带标记的切片",
    include_inherited_labels: "包含继承标签",
    run_query: "运行查询",
    scanning: "扫描中…",
    reset: "重置",
    active_filters: "当前过滤条件",
    clear_all: "清空",
    remove_filter: "移除过滤条件",
    extension_short: "扩展",
    current_scope_summary: "当前范围：{samples} 条已索引切片 · {files} 个候选文件 · 展示前 {limit} 条，排序方式：{sort}",
    matches_retained: "扫描 {rows} 行后保留了 {matches} 条匹配结果",
    total: "总计",
    dim_intent: "意图",
    dim_language: "语言",
    dim_domain: "领域",
    dim_task: "任务",
    dim_difficulty: "难度",
    dim_concept: "概念",
    dim_agentic: "代理式",
    dim_constraint: "约束",
    dim_context: "上下文",
    scope_kind_global: "全局",
    scope_kind_dir: "目录",
    scope_kind_file: "文件",
    thinking_fast: "快思考",
    thinking_slow: "慢思考",
  },
};

function detectInitialLocale() {
  try {
    const saved = window.localStorage.getItem("dashboard.locale");
    if (saved && ["en", "zh"].includes(saved)) return saved;
  } catch (error) {
    // Ignore storage failures.
  }
  const lang = String(window.navigator.language || "").toLowerCase();
  return lang.startsWith("zh") ? "zh" : "en";
}

STATE.locale = detectInitialLocale();

function loadSidebarPreference() {
  try {
    return window.localStorage.getItem("dashboard.sidebarCollapsed") === "1";
  } catch (error) {
    return false;
  }
}

function applySidebarState() {
  const shell = document.querySelector(".shell");
  const button = document.getElementById("sidebar-toggle");
  const collapsed = STATE.sidebarCollapsed;
  shell.classList.toggle("sidebar-collapsed", collapsed);
  button.textContent = collapsed ? ">" : "<";
  button.setAttribute("aria-expanded", collapsed ? "false" : "true");
  button.setAttribute("title", collapsed ? t("expand_navigator") : t("collapse_navigator"));
}

function toggleSidebar() {
  STATE.sidebarCollapsed = !STATE.sidebarCollapsed;
  try {
    window.localStorage.setItem("dashboard.sidebarCollapsed", STATE.sidebarCollapsed ? "1" : "0");
  } catch (error) {
    // Ignore storage failures; sidebar toggle should still work.
  }
  applySidebarState();
}

function setLocale(locale) {
  if (!["en", "zh"].includes(locale)) return;
  STATE.locale = locale;
  try {
    window.localStorage.setItem("dashboard.locale", locale);
  } catch (error) {
    // Ignore storage failures.
  }
  persistDashboardState();
  renderChromeText();
  renderHero();
  renderTree();
  renderScope();
}

const DIM_COLORS = {
  intent: "#2563eb",
  language: "#7c3aed",
  domain: "#0f766e",
  task: "#c2410c",
  difficulty: "#15803d",
  concept: "#be185d",
  agentic: "#9a3412",
  constraint: "#4f46e5",
  context: "#0d9488",
};

const HIST_COLORS = {
  value_score: "#2563eb",
  value_score_v2: "#1d4ed8",
  complexity_overall: "#c2410c",
  quality_overall: "#15803d",
  reasoning_overall: "#7c3aed",
  rarity_score: "#0f766e",
  extension_rarity_score: "#0891b2",
  rarity_v2_score: "#0ea5a4",
  selection_score: "#be185d",
  selection_score_v2: "#db2777",
};

const HIST_LABELS = {
  value_score: "value_score",
  value_score_v2: "value_score_v2",
  complexity_overall: "complexity",
  quality_overall: "quality",
  reasoning_overall: "reasoning",
  rarity_score: "median_rarity",
  extension_rarity_score: "extension_rarity_preview",
  rarity_v2_score: "rarity_v2",
  selection_score: "selection",
  selection_score_v2: "selection_v2",
};

const KEEP_RATE_THRESHOLDS = ["4.0", "5.0", "6.0", "7.0"];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function t(key, vars = null) {
  const table = I18N[STATE.locale] || I18N.en;
  const fallback = I18N.en[key] ?? key;
  const template = table[key] ?? fallback;
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (_match, name) => String(vars[name] ?? ""));
}

function hasTranslation(key) {
  return Object.prototype.hasOwnProperty.call(I18N.en, key) || Object.prototype.hasOwnProperty.call(I18N.zh, key);
}

function prettifyKey(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function dimensionLabel(dim) {
  const key = `dim_${dim}`;
  return hasTranslation(key) ? t(key) : prettifyKey(dim);
}

function scopeKindLabel(kind) {
  const key = `scope_kind_${kind}`;
  return hasTranslation(key) ? t(key) : prettifyKey(kind);
}

function turnKindLabel(kind) {
  const key = `turn_kind_${kind}`;
  return hasTranslation(key) ? t(key) : String(kind || "");
}

function thinkingModeLabel(mode) {
  const key = `thinking_${mode}`;
  return hasTranslation(key) ? t(key) : String(mode || "");
}

function localizedUnitLabel(label) {
  if (label === "samples") return t("samples");
  if (label === "units") return t("units");
  return label || t("items");
}

function renderChromeText() {
  document.documentElement.lang = STATE.locale === "zh" ? "zh-CN" : "en";
  const sidebarTitle = document.getElementById("sidebar-title");
  const sidebarSub = document.getElementById("sidebar-sub");
  const search = document.getElementById("scope-search");
  const goParent = document.getElementById("go-parent");
  const goGlobal = document.getElementById("go-global");
  if (sidebarTitle) sidebarTitle.textContent = t("scope_navigator");
  if (sidebarSub) sidebarSub.textContent = t("scope_navigator_sub");
  if (search) search.placeholder = t("search_folders_files");
  const allBtn = document.querySelector('[data-filter="all"]');
  const dirBtn = document.querySelector('[data-filter="dir"]');
  const fileBtn = document.querySelector('[data-filter="file"]');
  if (allBtn) allBtn.textContent = t("filter_all");
  if (dirBtn) dirBtn.textContent = t("filter_folders");
  if (fileBtn) fileBtn.textContent = t("filter_files");
  if (goParent) goParent.textContent = t("up_one_level");
  if (goGlobal) goGlobal.textContent = t("back_to_global");
  applySidebarState();
}

function scoreClass(value) {
  if (value >= 8) return "value-strong";
  if (value >= 6) return "value-mid";
  if (value >= 4) return "value-warn";
  return "value-low";
}

function keepRateClass(value) {
  if (value >= 0.75) return "keep-rate-strong";
  if (value >= 0.5) return "keep-rate-mid";
  if (value >= 0.25) return "keep-rate-warn";
  return "keep-rate-low";
}

function normalizedKeepRateThreshold(value) {
  const text = String(value || "");
  return KEEP_RATE_THRESHOLDS.includes(text) ? text : "7.0";
}

function fileKeepRateValue(row, threshold = STATE.keepRateThreshold) {
  const key = normalizedKeepRateThreshold(threshold);
  if (row && row.keep_rates && Object.prototype.hasOwnProperty.call(row.keep_rates, key)) {
    return Number(row.keep_rates[key]) || 0;
  }
  if (key === "7.0" && row && row.keep_rate_7 !== undefined && row.keep_rate_7 !== null) {
    return Number(row.keep_rate_7) || 0;
  }
  return 0;
}

function keepRateLabel(threshold = STATE.keepRateThreshold) {
  const value = normalizedKeepRateThreshold(threshold);
  return `${t("keep_rate_threshold")} ≥${Number(value).toFixed(0)}`;
}

function hexToRgba(hex, alpha) {
  const value = String(hex || "").replace("#", "").trim();
  if (value.length !== 6) return `rgba(37, 99, 235, ${alpha})`;
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function confColor(value) {
  if (value >= 0.9) return "#d1fae5";
  if (value >= 0.8) return "#fef3c7";
  if (value >= 0.7) return "#fed7aa";
  return "#fecaca";
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function fmtInt(value) {
  return Number(value || 0).toLocaleString();
}

function parseNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function textPreview(value) {
  return escapeHtml(String(value || "").trim() || "-");
}

function truncateMiddle(value, maxLen) {
  const text = String(value || "");
  if (!maxLen || text.length <= maxLen) return text;
  const head = Math.max(Math.floor((maxLen - 1) * 0.65), 1);
  const tail = Math.max(maxLen - head - 1, 1);
  return `${text.slice(0, head)}…${text.slice(-tail)}`;
}

function escapeAttrJson(value) {
  return escapeHtml(JSON.stringify(value));
}

function splitCsv(value) {
  return String(value || "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function mergeCsvValues(existing, additions) {
  const seen = new Set();
  const merged = [];
  for (const part of [...splitCsv(existing), ...splitCsv(additions)]) {
    const key = part.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(part);
  }
  return merged.join(", ");
}

function removeCsvValue(existing, removal) {
  const needle = String(removal || "").trim().toLowerCase();
  return splitCsv(existing)
    .filter((part) => part.toLowerCase() !== needle)
    .join(", ");
}

function parseExtensionQueryToken(token) {
  const raw = String(token || "").trim();
  let match = /^extid:([^:]+)$/.exec(raw);
  if (match) return { kind: "extension", extensionId: match[1] };
  match = /^extfield:([^:]+):([^:]+)$/.exec(raw);
  if (match) return { kind: "field", extensionId: match[1], field: match[2] };
  match = /^ext:([^:]+):([^:]+):(.+)$/.exec(raw);
  if (!match) return null;
  return { kind: "value", extensionId: match[1], field: match[2], value: match[3] };
}

function formatExplorerQueryToken(token) {
  const parsed = parseExtensionQueryToken(token);
  if (!parsed) return String(token || "");
  if (parsed.kind === "extension") {
    return `${t("extension_short_id")} · ${parsed.extensionId}`;
  }
  if (parsed.kind === "field") {
    return `${t("extension_field_short")} · ${parsed.extensionId} / ${parsed.field}`;
  }
  return `${t("extension_short")} · ${parsed.extensionId} / ${parsed.field} = ${parsed.value}`;
}

function explorerRegistry() {
  if (!window.__SFT_DASHBOARD_EXPLORER__) {
    window.__SFT_DASHBOARD_EXPLORER__ = { preview: {}, detail: {} };
  }
  return window.__SFT_DASHBOARD_EXPLORER__;
}

function persistDashboardState() {
  try {
    const state = {
      scope: STATE.currentId,
      locale: STATE.locale,
      tagBarMode: STATE.tagBarMode,
      aggregationMode: STATE.aggregationMode,
      keepRateThreshold: STATE.keepRateThreshold,
      explorer: {
        sort: STATE.explorer.sort,
        query: STATE.explorer.queryModel,
      },
    };
    const encoded = encodeURIComponent(JSON.stringify(state));
    const nextHash = `dashboard=${encoded}`;
    if (window.location.hash.slice(1) !== nextHash) {
      window.location.hash = nextHash;
    }
  } catch (error) {
    // Ignore hash persistence failures.
  }
}

function restoreDashboardState() {
  try {
    const raw = window.location.hash.replace(/^#/, "");
    if (!raw.startsWith("dashboard=")) return;
    const payload = JSON.parse(decodeURIComponent(raw.slice("dashboard=".length)));
    if (payload.scope && DATA.scopes[payload.scope]) {
      STATE.currentId = payload.scope;
    }
    if (payload.locale && ["en", "zh"].includes(payload.locale)) {
      STATE.locale = payload.locale;
    }
    if (payload.tagBarMode && ["hidden", "relative", "global-log"].includes(payload.tagBarMode)) {
      STATE.tagBarMode = payload.tagBarMode;
    }
    if (payload.aggregationMode && ["sample", "conversation"].includes(payload.aggregationMode)) {
      STATE.aggregationMode = payload.aggregationMode;
    }
    if (payload.keepRateThreshold) {
      STATE.keepRateThreshold = normalizedKeepRateThreshold(payload.keepRateThreshold);
    }
    if (payload.explorer && typeof payload.explorer === "object") {
      if (payload.explorer.sort) STATE.explorer.sort = payload.explorer.sort;
      if (payload.explorer.query && typeof payload.explorer.query === "object") {
        STATE.explorer.queryModel = {
          ...STATE.explorer.queryModel,
          ...payload.explorer.query,
        };
      }
    }
    let cursor = getScope(STATE.currentId);
    while (cursor && cursor.parent_id) {
      STATE.expanded.add(cursor.parent_id);
      cursor = getScope(cursor.parent_id);
    }
  } catch (error) {
    // Ignore malformed hashes.
  }
}

function getScope(id) {
  return DATA.scopes[id];
}

async function ensureScopeDetail(id) {
  const scope = getScope(id);
  if (!scope || !scope.detail_path || scope.__detailLoaded) return scope;
  if (SCOPE_DETAIL_CACHE.has(id)) {
    await SCOPE_DETAIL_CACHE.get(id);
    return getScope(id);
  }

  const promise = (async () => {
    let detail = null;
    if (scope.detail_script_path) {
      await ensureDashboardScriptLoaded(scope.detail_script_path);
      detail = dashboardDataRegistry().scopes[id] || null;
    }
    if (!detail) {
      detail = await fetchDashboardJson(scope.detail_path);
    }
    if (detail && typeof detail === "object") {
      Object.assign(scope, detail);
    }
    scope.__detailLoaded = true;
  })();
  SCOPE_DETAIL_CACHE.set(id, promise);
  try {
    await promise;
  } finally {
    SCOPE_DETAIL_CACHE.delete(id);
  }
  return scope;
}

function getCurrentScope() {
  return getScope(STATE.currentId) || getScope(DATA.root_id);
}

function scopeSummary(scope) {
  const modes = (scope || {}).summary_modes || {};
  return modes[STATE.aggregationMode] || (scope || {}).summary || {};
}

function aggregationPayload(bucket) {
  if (!bucket) return null;
  return (bucket.modes || {})[STATE.aggregationMode] || bucket;
}

function aggregationUnitLabel(bucket, fallback = "items") {
  const payload = aggregationPayload(bucket) || {};
  return localizedUnitLabel(payload.unit_label || fallback);
}

function aggregationInfoHtml() {
  const body = escapeHtml(t("aggregation_help_body")).replaceAll("\n", "<br>");
  return `<div class="info-inline" id="aggregation-info-wrap"><button class="info-btn" id="aggregation-info-btn" type="button" title="${escapeHtml(t("aggregation_help_title"))}">!</button><div class="info-pop" id="aggregation-info-pop"><strong>${escapeHtml(t("aggregation_help_title"))}</strong><br>${body}</div></div>`;
}

function resetExplorerView(resetQuery = false) {
  STATE.explorer.runId += 1;
  STATE.explorer.busy = false;
  STATE.explorer.scanned = 0;
  STATE.explorer.matched = 0;
  STATE.explorer.scopeTotal = 0;
  STATE.explorer.scopeDone = 0;
  STATE.explorer.status = "";
  STATE.explorer.results = [];
  STATE.explorer.detailDocId = "";
  STATE.explorer.detailScopeId = "";
  STATE.explorer.detailData = null;
  STATE.explorer.detailLoading = false;
  if (resetQuery) {
    STATE.explorer.queryModel = {
      tagQuery: "",
      language: "",
      intent: "",
      sourcePath: "",
      textQuery: "",
      flagQuery: "",
      convValueMin: "",
      convSelectionMin: "",
      convSelectionMax: "",
      peakComplexityMin: "",
      turnCountMin: "",
      observedTurnRatioMin: "",
      observedTurnRatioMax: "",
      rarityConfidenceMin: "",
      rarityConfidenceMax: "",
      minValue: "",
      maxValue: "",
      maxQuality: "",
      minSelection: "",
      maxSelection: "",
      maxConfidence: "",
      thinkingMode: "",
      hasFlags: false,
      includeInherited: false,
    };
    STATE.explorer.sort = "quality_asc";
  }
}

function sourcePathMatchesScope(scope, sourcePath) {
  const term = String(sourcePath || "").trim().toLowerCase();
  if (!term) return true;
  const haystack = `${scope.path || ""} ${scope.label || ""}`.toLowerCase();
  return haystack.includes(term);
}

function explorerFileScopeIds(scope, queryModel = STATE.explorer.queryModel) {
  if (!scope) return [];
  if (scope.explorer) return sourcePathMatchesScope(scope, (queryModel || {}).sourcePath) ? [scope.id] : [];
  return (scope.descendant_files || [])
    .map((path) => `file:${path}`)
    .filter((scopeId) => {
      const item = getScope(scopeId);
      return item && item.explorer && sourcePathMatchesScope(item, (queryModel || {}).sourcePath);
    });
}

function explorerTotalSamples(scope) {
  return explorerFileScopeIds(scope, {}).reduce((sum, scopeId) => {
    const explorer = (getScope(scopeId) || {}).explorer || {};
    return sum + Number(explorer.sample_count || 0);
  }, 0);
}

function scopeSupportsExplorer(scope) {
  return !!(DATA.explorer && DATA.explorer.enabled && explorerFileScopeIds(scope, {}).length);
}

function scopeHasPass2Data(scope) {
  return explorerFileScopeIds(scope, {}).some((scopeId) => {
    const item = getScope(scopeId) || {};
    return !!(item.pass2 || item.has_pass2 || ((item.summary || {}).scored_total > 0));
  });
}

function compactTreeMeta(summary) {
  const bits = [];
  if (summary.turn_kind) bits.push(turnKindLabel(summary.turn_kind));
  const sampleCount = Number(summary.scored_total || summary.pass1_total || 0);
  if (sampleCount) bits.push(`N ${fmtInt(sampleCount)}`);
  if (summary.mean_value !== null && summary.mean_value !== undefined) {
    bits.push(`V ${fmt(summary.mean_value, 1)}`);
  }
  if (summary.mean_selection !== null && summary.mean_selection !== undefined) {
    bits.push(`S ${fmt(summary.mean_selection, 1)}`);
  }
  return bits;
}

function toggleExpanded(id) {
  if (STATE.expanded.has(id)) STATE.expanded.delete(id);
  else STATE.expanded.add(id);
  renderTree();
}

async function setScope(id) {
  if (!DATA.scopes[id]) return;
  if (STATE.currentId !== id) resetExplorerView(false);
  STATE.currentId = id;
  let cursor = DATA.scopes[id];
  while (cursor && cursor.parent_id) {
    STATE.expanded.add(cursor.parent_id);
    cursor = DATA.scopes[cursor.parent_id];
  }
  await ensureScopeDetail(id);
  persistDashboardState();
  await render();
}

function matchesFilter(scope) {
  if (!scope) return false;
  if (STATE.filter === "all") return true;
  if (STATE.filter === "dir") return scope.kind === "dir" || scope.kind === "global";
  return scope.kind === "file";
}

function matchesQuery(scope) {
  if (!STATE.query) return true;
  const haystack = `${scope.label} ${scope.path}`.toLowerCase();
  return haystack.includes(STATE.query);
}

function subtreeMatches(id) {
  const scope = getScope(id);
  if (!scope) return false;
  if (matchesFilter(scope) && matchesQuery(scope)) return true;
  return (scope.children || []).some(subtreeMatches);
}

function renderTreeNode(id, depth = 0) {
  const scope = getScope(id);
  if (!scope || !subtreeMatches(id)) return "";

  const isExpanded = STATE.expanded.has(id) || id === DATA.root_id;
  const hasChildren = (scope.children || []).length > 0;
  const summary = scopeSummary(scope);
  const kindClass = `kind-${scope.kind === "global" ? "global" : (scope.kind === "dir" ? "dir" : "file")}`;
  const metaBits = compactTreeMeta(summary);

  let html = `<div class="tree-node ${isExpanded ? "expanded" : ""}">`;
  html += `<div class="tree-row ${STATE.currentId === id ? "active" : ""}" data-scope="${escapeHtml(id)}">`;
  html += `<span class="tree-indent" style="margin-left:${depth * 6}px"></span>`;
  if (hasChildren) {
    html += `<span class="tree-toggle" data-toggle="${escapeHtml(id)}">${isExpanded ? "▾" : "▸"}</span>`;
  } else {
    html += `<span class="tree-toggle"></span>`;
  }
  html += `<div class="tree-content">`;
  html += `<div class="tree-main">`;
  html += `<span class="tree-kind-dot ${kindClass}" title="${escapeHtml(scopeKindLabel(scope.kind))}"></span>`;
  html += `<span class="tree-label" title="${escapeHtml(scope.path || scope.label)}">${escapeHtml(scope.label)}</span>`;
  html += `</div>`;
  if (metaBits.length) {
    html += `<div class="tree-sub">${metaBits.map((bit) => `<span class="tree-meta-pill">${escapeHtml(bit)}</span>`).join("")}</div>`;
  }
  html += `</div>`;
  html += `</div>`;
  if (hasChildren) {
    html += `<div class="tree-children">`;
    for (const childId of scope.children) {
      html += renderTreeNode(childId, depth + 1);
    }
    html += `</div>`;
  }
  html += `</div>`;
  return html;
}

function renderTree() {
  const tree = document.getElementById("tree");
  tree.innerHTML = renderTreeNode(DATA.root_id);

  tree.querySelectorAll("[data-scope]").forEach((node) => {
    node.addEventListener("click", (event) => {
      const toggleId = event.target.getAttribute("data-toggle");
      if (toggleId) {
        toggleExpanded(toggleId);
        event.stopPropagation();
        return;
      }
      const scopeId = node.getAttribute("data-scope");
      const scope = getScope(scopeId);
      if (scope && (scope.children || []).length > 0 && STATE.currentId === scopeId) {
        toggleExpanded(scopeId);
      }
      setScope(scopeId);
    });
  });
  tree.querySelectorAll("[data-toggle]").forEach((node) => {
    node.addEventListener("click", (event) => {
      toggleExpanded(node.getAttribute("data-toggle"));
      event.stopPropagation();
    });
  });
}

function renderHero() {
  document.getElementById("hero-title").textContent = DATA.title_key ? t(DATA.title_key) : (DATA.title || t("dashboard_title_generic"));
  document.getElementById("hero-subtitle").textContent = DATA.subtitle || "";
  document.title = DATA.title_key ? t(DATA.title_key) : (DATA.title || t("dashboard_title_generic"));

  const root = getScope(DATA.root_id);
  const summary = root ? scopeSummary(root) : {};
  const heroStats = [
    [t("scopes"), Object.keys(DATA.scopes || {}).length],
    [t("files"), summary.file_count || 0],
  ];
  if (summary.pass1_total) heroStats.push([t("labeled"), summary.pass1_total]);
  if (summary.scored_total) heroStats.push([t("scored"), summary.scored_total]);
  if (summary.mean_value !== null && summary.mean_value !== undefined) heroStats.push([t("mean_value"), fmt(summary.mean_value, 1)]);
  document.getElementById("hero-stats").innerHTML = heroStats
    .map(([label, value]) => `<div class="pill"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
  document.getElementById("hero-toolbar").innerHTML = renderTagBarControlsInline();
}

function breadcrumbIds(scope) {
  const ids = [];
  let cursor = scope;
  while (cursor) {
    ids.push(cursor.id);
    cursor = cursor.parent_id ? getScope(cursor.parent_id) : null;
  }
  return ids.reverse();
}

function renderBreadcrumbs(scope) {
  const parts = breadcrumbIds(scope).map((id, index, items) => {
    const item = getScope(id);
    if (index === items.length - 1) return `<span>${escapeHtml(item.label)}</span>`;
    return `<button class="crumb-btn" data-crumb="${escapeHtml(id)}">${escapeHtml(item.label)}</button><span>/</span>`;
  });
  document.getElementById("breadcrumbs").innerHTML = parts.join("");
  document.querySelectorAll("[data-crumb]").forEach((node) => {
    node.addEventListener("click", () => setScope(node.getAttribute("data-crumb")));
  });
}

function section(title, body, meta = "", open = true) {
  return `<details class="section-card" ${open ? "open" : ""}>
    <summary class="section-summary">
      <span class="section-title">${escapeHtml(title)}</span>
      <span class="section-meta">${escapeHtml(meta)}</span>
    </summary>
    <div class="section-body">${body}</div>
  </details>`;
}

function renderCards(cards) {
  if (!cards.length) return "";
  return `<div class="cards">${
    cards.map((card) => `<div class="card">
      <div class="card-label">${escapeHtml(card.label)}</div>
      <div class="card-value ${card.className || ""}">${escapeHtml(card.value)}</div>
      ${card.sub ? `<div class="card-sub">${escapeHtml(card.sub)}</div>` : ""}
    </div>`).join("")
  }</div>`;
}

function renderBarChart(items, color, maxValue = null, actionBuilder = null) {
  if (!items || !items.length) return `<div class="empty">${escapeHtml(t("no_data"))}</div>`;
  const ceiling = maxValue || Math.max(...items.map((item) => Number(item.value) || 0), 1);
  return items.map((item) => {
    const pct = ceiling > 0 ? ((Number(item.value) || 0) / ceiling) * 100 : 0;
    const labelHtml = actionBuilder
      ? `<button class="link-btn bar-label" type="button" data-explorer-patch="${escapeAttrJson(actionBuilder(item))}" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</button>`
      : `<div class="bar-label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</div>`;
    return `<div class="bar-row">
      ${labelHtml}
      <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%;background:${color}"></div></div>
      <div class="bar-value">${escapeHtml(item.display || item.value)}</div>
    </div>`;
  }).join("");
}

function renderHistogram(label, bins, color, stats, bucketActionBuilder = null) {
  const max = Math.max(...bins, 1);
  const bars = bins.map((count, index) => {
    const height = Math.max((count / max) * 100, 2);
    const patch = bucketActionBuilder ? ` data-explorer-patch="${escapeAttrJson(bucketActionBuilder(index + 1, count))}"` : "";
    const role = bucketActionBuilder ? "button" : "";
    const cls = bucketActionBuilder ? "hist-bar link-btn" : "hist-bar";
    return `<div class="${cls}"${patch} style="height:${height}%;background:${color}" title="${index + 1}: ${count}" ${role ? `role="${role}"` : ""}></div>`;
  }).join("");
  const labels = bins.map((_, index) => `<span>${index + 1}</span>`).join("");
  const statLine = stats ? `<div class="note">${escapeHtml(t("mean"))}=${fmt(stats.mean, 2)} · ${escapeHtml(t("min"))}=${fmt(stats.min, 1)} · ${escapeHtml(t("max"))}=${fmt(stats.max, 1)}</div>` : "";
  return `<div class="mini-panel">
    <h4>${escapeHtml(label)}</h4>
    ${statLine}
    <div class="hist-row">${bars}</div>
    <div class="hist-labels">${labels}</div>
  </div>`;
}

function renderRankingTable(items, metricLabel, valueRenderer, actionBuilder = null, options = null) {
  if (!items || !items.length) return `<div class="empty">${escapeHtml(t("no_data"))}</div>`;
  const rankOptions = options || {};
  const barMetricKey = rankOptions.barMetricKey || "";
  const barColor = rankOptions.barColor || "#2563eb";
  const barMetricLabel = rankOptions.barMetricLabel || t("count");
  const globalBarMax = Number(rankOptions.globalBarMax) || 0;
  const barValues = barMetricKey
    ? items.map((item) => Number(item?.[barMetricKey]) || 0)
    : [];
  const localBarMax = barValues.length ? Math.max(...barValues, 1) : 0;
  const barMax = STATE.tagBarMode === "global-log" && globalBarMax > 0 ? globalBarMax : localBarMax;
  const rows = items.map((item, index) => `<tr>
    <td class="rank-cell">${index + 1}</td>
    <td class="rank-tag-cell">${renderRankingTagCell(item, actionBuilder, {
      barMetricKey,
      barMetricLabel,
      barMax,
      barColor,
      buttonDataAttr: rankOptions.buttonDataAttr,
    })}</td>
    <td class="metric-cell">${escapeHtml(valueRenderer(item))}</td>
  </tr>`).join("");
  return `<table class="data-table compact-table">
    <thead><tr><th>#</th><th>${escapeHtml(t("tag"))}</th><th>${escapeHtml(metricLabel)}</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderRankingTagCell(item, actionBuilder, options = null) {
  const rankOptions = options || {};
  const metricKey = rankOptions.barMetricKey || "";
  const rawMetric = metricKey ? Number(item?.[metricKey]) || 0 : 0;
  const barMax = Number(rankOptions.barMax) || 0;
  const extraButtonAttr = rankOptions.buttonDataAttr ? ` ${rankOptions.buttonDataAttr}="1"` : "";
  const mode = STATE.tagBarMode || "relative";
  let pct = 0;
  if (metricKey && barMax > 0 && mode !== "hidden") {
    if (mode === "global-log") {
      pct = (Math.log1p(Math.max(rawMetric, 0)) / Math.log1p(barMax)) * 100;
    } else {
      pct = (rawMetric / barMax) * 100;
    }
    pct = Math.max(6, pct);
  }
  const fill = metricKey && mode !== "hidden"
    ? `<div class="rank-tag-bar" style="width:${pct.toFixed(1)}%;background:linear-gradient(90deg, ${hexToRgba(rankOptions.barColor, 0.18)} 0%, ${hexToRgba(rankOptions.barColor, 0.10)} 85%, ${hexToRgba(rankOptions.barColor, 0.04)} 100%);border:1px solid ${hexToRgba(rankOptions.barColor, 0.12)}" title="${escapeHtml(`${item.label} · ${rankOptions.barMetricLabel || t("count")}: ${rawMetric}`)}"></div>`
    : "";
  const labelHtml = actionBuilder
    ? `<button class="link-btn rank-tag-link" type="button"${extraButtonAttr} data-explorer-patch="${escapeAttrJson(actionBuilder(item))}" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</button>`
    : `<span class="rank-tag-text" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span>`;
  return `<div class="rank-tag-wrap">${fill}<div class="rank-tag-content">${labelHtml}</div></div>`;
}

const FILE_RANKING_COLUMNS = {
  file: {labelKey: "source_file", defaultDirection: "asc"},
  count: {labelKey: "count", defaultDirection: "desc"},
  mean_value: {labelKey: "value", defaultDirection: "desc"},
  mean_complexity: {labelKey: "complexity", defaultDirection: "desc"},
  mean_quality: {labelKey: "quality", defaultDirection: "desc"},
  mean_rarity: {labelKey: "mean_rarity", defaultDirection: "desc"},
  mean_selection: {labelKey: "selection", defaultDirection: "desc"},
  keep_rate_dynamic: {labelKey: "keep_rate_7", defaultDirection: "desc"},
  mean_turns: {labelKey: "mean_turns", defaultDirection: "desc"},
};

function fileRankingIndicator(key) {
  if (STATE.fileRankingSort.key !== key) return "↕";
  return STATE.fileRankingSort.direction === "asc" ? "↑" : "↓";
}

function renderFileRankingHeader(key) {
  const column = FILE_RANKING_COLUMNS[key];
  const active = STATE.fileRankingSort.key === key;
  const label = key === "keep_rate_dynamic" ? keepRateLabel() : t(column.labelKey);
  return `<th><button class="th-sort-btn ${active ? "active" : ""}" data-file-sort="${escapeHtml(key)}" type="button">
    <span>${escapeHtml(label)}</span>
    <span class="sort-indicator">${fileRankingIndicator(key)}</span>
  </button></th>`;
}

function sortPerFileSummary(rows) {
  const {key, direction} = STATE.fileRankingSort;
  const factor = direction === "asc" ? 1 : -1;
  return rows.slice().sort((left, right) => {
    if (key === "file") {
      const cmp = String(left.file || "").localeCompare(String(right.file || ""), undefined, {
        numeric: true,
        sensitivity: "base",
      });
      if (cmp !== 0) return cmp * factor;
    } else if (key === "keep_rate_dynamic") {
      const delta = fileKeepRateValue(left) - fileKeepRateValue(right);
      if (delta !== 0) return delta * factor;
    } else {
      const delta = (Number(left[key]) || 0) - (Number(right[key]) || 0);
      if (delta !== 0) return delta * factor;
    }
    return String(left.file || "").localeCompare(String(right.file || ""), undefined, {
      numeric: true,
      sensitivity: "base",
    });
  });
}

function toggleFileRankingSort(key) {
  const column = FILE_RANKING_COLUMNS[key];
  if (!column) return;
  if (STATE.fileRankingSort.key === key) {
    STATE.fileRankingSort.direction = STATE.fileRankingSort.direction === "desc" ? "asc" : "desc";
  } else {
    STATE.fileRankingSort = {key, direction: column.defaultDirection};
  }
  renderScope();
}

function explorerAssetPath(scopeExplorer, relpath) {
  return `${scopeExplorer.assets_dir}/${relpath}`;
}

function ensureExplorerScriptLoaded(path) {
  const cached = EXPLORER_CACHE.scripts.get(path);
  if (cached) return cached.promise;
  const script = document.createElement("script");
  const entry = { node: script, promise: null };
  entry.promise = new Promise((resolve, reject) => {
    script.src = path;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load explorer asset: ${path}`));
    document.head.appendChild(script);
  });
  EXPLORER_CACHE.scripts.set(path, entry);
  return entry.promise;
}

function dropExplorerScript(path) {
  const entry = EXPLORER_CACHE.scripts.get(path);
  if (entry && entry.node && entry.node.parentNode) {
    entry.node.parentNode.removeChild(entry.node);
  }
  EXPLORER_CACHE.scripts.delete(path);
}

function unloadScopePreview(scopeId) {
  const assets = EXPLORER_CACHE.previewAssets.get(scopeId) || [];
  const registry = explorerRegistry();
  for (const asset of assets) {
    delete registry.preview[asset.key];
    dropExplorerScript(asset.path);
  }
  EXPLORER_CACHE.previewAssets.delete(scopeId);
  EXPLORER_CACHE.previews.delete(scopeId);
}

function unloadDetailChunk(cacheKey) {
  const asset = EXPLORER_CACHE.detailAssets.get(cacheKey);
  const registry = explorerRegistry();
  if (asset) {
    delete registry.detail[asset.chunkKey];
    dropExplorerScript(asset.path);
  }
  EXPLORER_CACHE.detailAssets.delete(cacheKey);
  EXPLORER_CACHE.details.delete(cacheKey);
}

function retainScopePreview(scopeId, rows, assets) {
  if (EXPLORER_CACHE.previews.has(scopeId)) {
    EXPLORER_CACHE.previews.delete(scopeId);
  }
  EXPLORER_CACHE.previews.set(scopeId, rows);
  EXPLORER_CACHE.previewAssets.set(scopeId, assets);
  while (EXPLORER_CACHE.previews.size > EXPLORER_PREVIEW_CACHE_LIMIT) {
    const oldestScopeId = EXPLORER_CACHE.previews.keys().next().value;
    if (!oldestScopeId) break;
    unloadScopePreview(oldestScopeId);
  }
}

function maybeRenderExplorerProgress(force = false) {
  const now = Date.now();
  if (!force && (now - EXPLORER_CACHE.lastProgressRender) < EXPLORER_PROGRESS_RENDER_MS) return;
  EXPLORER_CACHE.lastProgressRender = now;
  renderScope();
}

async function processScopePreviewRows(scopeId, visitRow) {
  if (EXPLORER_CACHE.previews.has(scopeId)) {
    for (const row of EXPLORER_CACHE.previews.get(scopeId) || []) {
      if (visitRow(row) === false) return false;
    }
    return true;
  }
  const scope = getScope(scopeId);
  const explorer = (scope || {}).explorer;
  if (!explorer) return true;
  const registry = explorerRegistry();
  const chunks = explorer.preview_chunks || [];
  const cacheable = chunks.length <= 3;
  const rows = [];
  const assets = [];
  for (const chunk of chunks) {
    const path = explorerAssetPath(explorer, chunk.path);
    await ensureExplorerScriptLoaded(path);
    const chunkRows = registry.preview[chunk.key] || [];
    if (cacheable) {
      rows.push(...chunkRows);
      assets.push({ key: chunk.key, path });
    }
    for (const row of chunkRows) {
      if (visitRow(row) === false) {
        if (!cacheable) {
          delete registry.preview[chunk.key];
          dropExplorerScript(path);
        }
        return false;
      }
    }
    if (!cacheable) {
      delete registry.preview[chunk.key];
      dropExplorerScript(path);
    }
  }
  if (cacheable) retainScopePreview(scopeId, rows, assets);
  return true;
}

async function loadExplorerDetail(scopeId, detailChunkKey, docId) {
  const cacheKey = `${scopeId}|${detailChunkKey}`;
  if (!EXPLORER_CACHE.details.has(cacheKey)) {
    const scope = getScope(scopeId);
    const explorer = (scope || {}).explorer;
    const chunk = (explorer && (explorer.detail_chunks || []).find((item) => item.key === detailChunkKey)) || null;
    if (!explorer || !chunk) return null;
    const path = explorerAssetPath(explorer, chunk.path);
    await ensureExplorerScriptLoaded(path);
    const registry = explorerRegistry();
    EXPLORER_CACHE.details.set(cacheKey, registry.detail[detailChunkKey] || {});
    EXPLORER_CACHE.detailAssets.set(cacheKey, { chunkKey: detailChunkKey, path });
    while (EXPLORER_CACHE.details.size > EXPLORER_DETAIL_CACHE_LIMIT) {
      const oldestCacheKey = EXPLORER_CACHE.details.keys().next().value;
      if (!oldestCacheKey || oldestCacheKey === cacheKey) break;
      unloadDetailChunk(oldestCacheKey);
    }
  } else {
    const chunkPayload = EXPLORER_CACHE.details.get(cacheKey);
    const asset = EXPLORER_CACHE.detailAssets.get(cacheKey);
    EXPLORER_CACHE.details.delete(cacheKey);
    EXPLORER_CACHE.detailAssets.delete(cacheKey);
    EXPLORER_CACHE.details.set(cacheKey, chunkPayload);
    if (asset) EXPLORER_CACHE.detailAssets.set(cacheKey, asset);
  }
  const chunkPayload = EXPLORER_CACHE.details.get(cacheKey) || {};
  return chunkPayload[docId] || null;
}

function parseExplorerQueryFromDom() {
  return {
    tagQuery: (document.getElementById("explorer-tag-query") || {}).value || "",
    language: (document.getElementById("explorer-language") || {}).value || "",
    intent: (document.getElementById("explorer-intent") || {}).value || "",
    sourcePath: (document.getElementById("explorer-source-path") || {}).value || "",
    textQuery: (document.getElementById("explorer-text-query") || {}).value || "",
    flagQuery: (document.getElementById("explorer-flag-query") || {}).value || "",
    convValueMin: (document.getElementById("explorer-conv-value-min") || {}).value || "",
    convSelectionMin: (document.getElementById("explorer-conv-selection-min") || {}).value || "",
    convSelectionMax: (document.getElementById("explorer-conv-selection-max") || {}).value || "",
    peakComplexityMin: (document.getElementById("explorer-peak-complexity-min") || {}).value || "",
    turnCountMin: (document.getElementById("explorer-turn-count-min") || {}).value || "",
    observedTurnRatioMin: (document.getElementById("explorer-observed-turn-ratio-min") || {}).value || "",
    observedTurnRatioMax: (document.getElementById("explorer-observed-turn-ratio-max") || {}).value || "",
    rarityConfidenceMin: (document.getElementById("explorer-rarity-confidence-min") || {}).value || "",
    rarityConfidenceMax: (document.getElementById("explorer-rarity-confidence-max") || {}).value || "",
    minValue: (document.getElementById("explorer-min-value") || {}).value || "",
    maxValue: (document.getElementById("explorer-max-value") || {}).value || "",
    maxQuality: (document.getElementById("explorer-max-quality") || {}).value || "",
    minSelection: (document.getElementById("explorer-min-selection") || {}).value || "",
    maxSelection: (document.getElementById("explorer-max-selection") || {}).value || "",
    maxConfidence: (document.getElementById("explorer-max-confidence") || {}).value || "",
    thinkingMode: (document.getElementById("explorer-thinking-mode") || {}).value || "",
    hasFlags: !!((document.getElementById("explorer-has-flags") || {}).checked),
    includeInherited: !!((document.getElementById("explorer-include-inherited") || {}).checked),
  };
}

function applyExplorerQueryPatch(patch) {
  const next = { ...STATE.explorer.queryModel };
  if (patch.tagQueryAppend) {
    next.tagQuery = mergeCsvValues(next.tagQuery, Array.isArray(patch.tagQueryAppend) ? patch.tagQueryAppend.join(", ") : patch.tagQueryAppend);
  }
  for (const [key, value] of Object.entries(patch || {})) {
    if (key === "tagQueryAppend") continue;
    next[key] = value;
  }
  STATE.explorer.queryModel = next;
  if (patch.sort) STATE.explorer.sort = patch.sort;
  persistDashboardState();
  renderScope();
}

function scrollExplorerIntoView() {
  const node = document.getElementById("explorer-panel");
  if (node && typeof node.scrollIntoView === "function") {
    node.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function rowMatchesExplorerQuery(row, query) {
  if (!row) return false;
  const tagTerms = String(query.tagQuery || "")
    .split(",")
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean);
  const rowTags = new Set([
    ...(row.flat_tags || []).map((item) => String(item).toLowerCase()),
    ...(row.extension_tags || []).map((item) => String(item).toLowerCase()),
  ]);
  if (tagTerms.length && !tagTerms.every((term) => rowTags.has(term))) return false;

  const language = String(query.language || "").trim().toLowerCase();
  if (language) {
    const langs = new Set((row.languages || []).map((item) => String(item).toLowerCase()));
    if (!langs.has(language)) return false;
  }

  const intent = String(query.intent || "").trim().toLowerCase();
  if (intent && String(row.intent || "").toLowerCase() !== intent) return false;

  const sourcePath = String(query.sourcePath || "").trim().toLowerCase();
  if (sourcePath) {
    const sourceHaystack = `${row.scope_path || ""} ${row.source_file || ""}`.toLowerCase();
    if (!sourceHaystack.includes(sourcePath)) return false;
  }

  const flagQuery = String(query.flagQuery || "").trim().toLowerCase();
  if (flagQuery) {
    const flags = new Set((row.flags || []).map((item) => String(item).toLowerCase()));
    if (!flags.has(flagQuery)) return false;
  }

  const minValue = parseNumber(query.minValue);
  if (minValue !== null && (row.value_score === null || row.value_score === undefined || Number(row.value_score) < minValue)) return false;
  const convValueMin = parseNumber(query.convValueMin);
  if (convValueMin !== null && (row.conv_value === null || row.conv_value === undefined || Number(row.conv_value) < convValueMin)) return false;
  const convSelectionMin = parseNumber(query.convSelectionMin);
  if (convSelectionMin !== null && (row.conv_selection === null || row.conv_selection === undefined || Number(row.conv_selection) < convSelectionMin)) return false;
  const convSelectionMax = parseNumber(query.convSelectionMax);
  if (convSelectionMax !== null && (row.conv_selection === null || row.conv_selection === undefined || Number(row.conv_selection) > convSelectionMax)) return false;
  const peakComplexityMin = parseNumber(query.peakComplexityMin);
  if (peakComplexityMin !== null && (row.peak_complexity === null || row.peak_complexity === undefined || Number(row.peak_complexity) < peakComplexityMin)) return false;
  const observedTurnRatioMin = parseNumber(query.observedTurnRatioMin);
  if (observedTurnRatioMin !== null && (row.observed_turn_ratio === null || row.observed_turn_ratio === undefined || Number(row.observed_turn_ratio) < observedTurnRatioMin)) return false;
  const observedTurnRatioMax = parseNumber(query.observedTurnRatioMax);
  if (observedTurnRatioMax !== null && (row.observed_turn_ratio === null || row.observed_turn_ratio === undefined || Number(row.observed_turn_ratio) > observedTurnRatioMax)) return false;
  const rarityConfidenceMin = parseNumber(query.rarityConfidenceMin);
  if (rarityConfidenceMin !== null && (row.rarity_confidence === null || row.rarity_confidence === undefined || Number(row.rarity_confidence) < rarityConfidenceMin)) return false;
  const rarityConfidenceMax = parseNumber(query.rarityConfidenceMax);
  if (rarityConfidenceMax !== null && (row.rarity_confidence === null || row.rarity_confidence === undefined || Number(row.rarity_confidence) > rarityConfidenceMax)) return false;
  const maxValue = parseNumber(query.maxValue);
  if (maxValue !== null && (row.value_score === null || row.value_score === undefined || Number(row.value_score) > maxValue)) return false;
  const maxQuality = parseNumber(query.maxQuality);
  if (maxQuality !== null && (row.quality_overall === null || row.quality_overall === undefined || Number(row.quality_overall) > maxQuality)) return false;
  const minSelection = parseNumber(query.minSelection);
  if (minSelection !== null && (row.selection_score === null || row.selection_score === undefined || Number(row.selection_score) < minSelection)) return false;
  const maxSelection = parseNumber(query.maxSelection);
  if (maxSelection !== null && (row.selection_score === null || row.selection_score === undefined || Number(row.selection_score) > maxSelection)) return false;
  const turnCountMin = parseNumber(query.turnCountMin);
  if (turnCountMin !== null && (row.turn_count === null || row.turn_count === undefined || Number(row.turn_count) < turnCountMin)) return false;
  const maxConfidence = parseNumber(query.maxConfidence);
  if (maxConfidence !== null && (row.confidence === null || row.confidence === undefined || Number(row.confidence) > maxConfidence)) return false;

  if (query.thinkingMode && String(row.thinking_mode || "") !== String(query.thinkingMode)) return false;
  if (!query.includeInherited && row.inherited) return false;
  if (query.hasFlags && !((row.flags || []).length > 0)) return false;

  const textQuery = String(query.textQuery || "").trim().toLowerCase();
  if (textQuery) {
    const haystack = [
      row.sample_id,
      row.query_preview,
      row.response_preview,
      row.source_file,
      row.scope_path,
    ].join(" ").toLowerCase();
    if (!haystack.includes(textQuery)) return false;
  }

  return true;
}

function compareExplorerRows(left, right, sortKey) {
  const numericSorts = {
    quality_asc: ["quality_overall", 1],
    value_asc: ["value_score", 1],
    confidence_asc: ["confidence", 1],
    selection_desc: ["selection_score", -1],
    conv_value_desc: ["conv_value", -1],
    conv_selection_desc: ["conv_selection", -1],
    turn_count_desc: ["turn_count", -1],
    observed_turn_ratio_asc: ["observed_turn_ratio", 1],
    observed_turn_ratio_desc: ["observed_turn_ratio", -1],
    rarity_confidence_asc: ["rarity_confidence", 1],
    rarity_confidence_desc: ["rarity_confidence", -1],
    rarity_desc: ["rarity_score", -1],
  };
  if (sortKey === "sample_id_asc") {
    return String(left.sample_id || "").localeCompare(String(right.sample_id || ""), undefined, { numeric: true, sensitivity: "base" });
  }
  const descriptor = numericSorts[sortKey] || numericSorts.quality_asc;
  const valueKey = descriptor[0];
  const direction = descriptor[1];
  const leftValue = Number(left[valueKey]);
  const rightValue = Number(right[valueKey]);
  const leftMissing = !Number.isFinite(leftValue);
  const rightMissing = !Number.isFinite(rightValue);
  if (leftMissing && rightMissing) {
    return String(left.sample_id || "").localeCompare(String(right.sample_id || ""), undefined, { numeric: true, sensitivity: "base" });
  }
  if (leftMissing) return 1;
  if (rightMissing) return -1;
  if (leftValue !== rightValue) return (leftValue - rightValue) * direction;
  return String(left.sample_id || "").localeCompare(String(right.sample_id || ""), undefined, { numeric: true, sensitivity: "base" });
}

function pushExplorerResult(results, row, sortKey, limit) {
  results.push(row);
  results.sort((left, right) => compareExplorerRows(left, right, sortKey));
  if (results.length > limit) results.length = limit;
}

async function runExplorerQuery() {
  const scope = getCurrentScope();
  if (!scopeSupportsExplorer(scope)) return;
  const query = parseExplorerQueryFromDom();
  const scopeIds = explorerFileScopeIds(scope, query);
  STATE.explorer.queryModel = query;
  STATE.explorer.results = [];
  STATE.explorer.detailDocId = "";
  STATE.explorer.detailScopeId = "";
  STATE.explorer.detailData = null;
  STATE.explorer.busy = true;
  STATE.explorer.scanned = 0;
  STATE.explorer.matched = 0;
  STATE.explorer.scopeTotal = scopeIds.length;
  STATE.explorer.scopeDone = 0;
  const runId = ++STATE.explorer.runId;
  EXPLORER_CACHE.lastProgressRender = 0;
  persistDashboardState();
  renderScope();

  for (let index = 0; index < scopeIds.length; index += 1) {
    if (runId !== STATE.explorer.runId) return;
    const scopeId = scopeIds[index];
    STATE.explorer.status = `Scanning ${index + 1}/${scopeIds.length}: ${getScope(scopeId).label}`;
    maybeRenderExplorerProgress();
    const completed = await processScopePreviewRows(scopeId, (row) => {
      if (runId !== STATE.explorer.runId) return false;
      STATE.explorer.scanned += 1;
      if (rowMatchesExplorerQuery(row, query)) {
        STATE.explorer.matched += 1;
        pushExplorerResult(STATE.explorer.results, row, STATE.explorer.sort, STATE.explorer.limit);
      }
      return true;
    });
    if (!completed && runId !== STATE.explorer.runId) return;
    STATE.explorer.scopeDone = index + 1;
    maybeRenderExplorerProgress();
  }

  if (runId !== STATE.explorer.runId) return;
  STATE.explorer.busy = false;
  STATE.explorer.status = t("status_done", { matches: fmtInt(STATE.explorer.matched), files: fmtInt(scopeIds.length) });
  persistDashboardState();
  maybeRenderExplorerProgress(true);
}

function setExplorerPreset(preset) {
  const next = {
    tagQuery: "",
    language: "",
    intent: "",
    sourcePath: "",
    textQuery: "",
    flagQuery: "",
    convValueMin: "",
    convSelectionMin: "",
    peakComplexityMin: "",
    turnCountMin: "",
    observedTurnRatioMin: "",
    observedTurnRatioMax: "",
    rarityConfidenceMin: "",
    rarityConfidenceMax: "",
    minValue: "",
    maxValue: "",
    maxQuality: "",
    minSelection: "",
    maxConfidence: "",
    thinkingMode: "",
    hasFlags: false,
    includeInherited: false,
    ...preset,
  };
  STATE.explorer.queryModel = next;
  if (preset.sort) STATE.explorer.sort = preset.sort;
  STATE.explorer.results = [];
  STATE.explorer.detailDocId = "";
  STATE.explorer.detailScopeId = "";
  STATE.explorer.detailData = null;
  persistDashboardState();
  renderScope();
}

async function openExplorerDetail(docId, scopeId, detailChunkKey) {
  STATE.explorer.detailDocId = docId;
  STATE.explorer.detailScopeId = scopeId;
  STATE.explorer.detailLoading = true;
  STATE.explorer.detailData = null;
  persistDashboardState();
  renderScope();
  const detail = await loadExplorerDetail(scopeId, detailChunkKey, docId);
  STATE.explorer.detailLoading = false;
  STATE.explorer.detailData = detail;
  persistDashboardState();
  renderScope();
}

function closeExplorerDetail() {
  STATE.explorer.detailDocId = "";
  STATE.explorer.detailScopeId = "";
  STATE.explorer.detailData = null;
  STATE.explorer.detailLoading = false;
  persistDashboardState();
  renderScope();
}

function renderDrawerTurns(conversations) {
  const turns = Array.isArray(conversations) ? conversations : [];
  if (!turns.length) return `<div class="note">${escapeHtml(t("no_conversation_turns"))}</div>`;
  const hiddenCount = Math.max(turns.length - EXPLORER_DRAWER_TURN_LIMIT, 0);
  const visibleTurns = turns.slice(0, EXPLORER_DRAWER_TURN_LIMIT).map((turn) => `<div class="conversation-turn">
      <div class="turn-role">${escapeHtml(turn.from || "unknown")}</div>
      <div class="turn-text">${escapeHtml(truncateMiddle(turn.value || "", EXPLORER_DRAWER_TEXT_LIMIT))}</div>
    </div>`).join("");
  return `<details class="drawer-collapse" ${turns.length <= EXPLORER_DRAWER_TURN_LIMIT ? "open" : ""}>
      <summary>${escapeHtml(t("conversation_preview", { shown: fmtInt(Math.min(turns.length, EXPLORER_DRAWER_TURN_LIMIT)), total: fmtInt(turns.length) }))}</summary>
      ${visibleTurns}
      ${hiddenCount ? `<div class="note drawer-truncate-note">${escapeHtml(t("preview_limited", { limit: fmtInt(EXPLORER_DRAWER_TURN_LIMIT), hidden: fmtInt(hiddenCount) }))}</div>` : ""}
    </details>`;
}

function renderDrawerJson(detail) {
  const raw = JSON.stringify(detail, null, 2);
  const truncated = truncateMiddle(raw, EXPLORER_DRAWER_JSON_LIMIT);
  return `<details class="drawer-collapse">
      <summary>${escapeHtml(raw.length > EXPLORER_DRAWER_JSON_LIMIT ? t("raw_json_preview_cap", { cap: fmtInt(EXPLORER_DRAWER_JSON_LIMIT) }) : t("raw_json_preview"))}</summary>
      <div class="drawer-json">${escapeHtml(truncated)}</div>
      ${raw.length > EXPLORER_DRAWER_JSON_LIMIT ? `<div class="note drawer-truncate-note">${escapeHtml(t("large_payload_truncated"))}</div>` : ""}
    </details>`;
}

function explorerPatchForTag(dim, label, sort = "quality_asc") {
  if (dim === "language") return { language: label, sort };
  if (dim === "intent") return { intent: label, sort };
  return { tagQueryAppend: [label], sort };
}

function renderChildren(scope) {
  const children = (scope.children || []).map(getScope).filter(Boolean);
  if (!children.length) return "";

  const rows = children.map((child) => {
    const summary = scopeSummary(child);
    const meanValue = summary.mean_value;
    const meanHtml = meanValue === null || meanValue === undefined
      ? "-"
      : `<span class="${scoreClass(meanValue)}">${fmt(meanValue, 1)}</span>`;
    return `<tr>
      <td><button class="link-btn" data-jump="${escapeHtml(child.id)}">${escapeHtml(child.label)}</button></td>
      <td><span class="kind-badge kind-${child.kind === "global" ? "global" : (child.kind === "dir" ? "dir" : "file")}">${escapeHtml(scopeKindLabel(child.kind))}</span></td>
      <td>${summary.file_count || 0}</td>
      <td>${summary.pass1_total || 0}</td>
      <td>${summary.scored_total || 0}</td>
      <td>${meanHtml}</td>
    </tr>`;
  }).join("");

  return section(
    t("children"),
    `<table class="data-table">
      <thead><tr><th>${escapeHtml(t("name"))}</th><th>${escapeHtml(t("type"))}</th><th>${escapeHtml(t("files"))}</th><th>${escapeHtml(t("labeled"))}</th><th>${escapeHtml(t("scored"))}</th><th>${escapeHtml(t("mean_value"))}</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`,
    t("direct_children", { count: children.length }),
    true,
  );
}

function renderTagBarControlsInline() {
  const tagButtons = [
    ["hidden", t("hide"), t("hide_tag_bars")],
    ["relative", t("normalized"), t("relative_hint")],
    ["global-log", t("global_log"), t("global_log_hint")],
  ].map(([value, label, hint]) => (
    `<button class="segmented-btn ${STATE.tagBarMode === value ? "active" : ""}" type="button" data-tag-bar-mode="${escapeHtml(value)}" title="${escapeHtml(hint)}">${escapeHtml(label)}</button>`
  )).join("");
  const aggButtons = [
    ["sample", t("agg_sample"), t("agg_sample_hint")],
    ["conversation", t("agg_conversation"), t("agg_conversation_hint")],
  ].map(([value, label, hint]) => (
    `<button class="segmented-btn ${STATE.aggregationMode === value ? "active" : ""}" type="button" data-aggregation-mode="${escapeHtml(value)}" title="${escapeHtml(hint)}">${escapeHtml(label)}</button>`
  )).join("");
  const localeButtons = [
    ["zh", t("chinese")],
    ["en", t("english")],
  ].map(([value, label]) => (
    `<button class="segmented-btn ${STATE.locale === value ? "active" : ""}" type="button" data-locale="${escapeHtml(value)}">${escapeHtml(label)}</button>`
  )).join("");
  return `<div class="hero-toolbar-group"><div class="hero-toolbar-label toolbar-inline-copy" title="${escapeHtml(t("language_toggle_help"))}">${escapeHtml(t("language"))}</div><div class="hero-toolbar-body"><div class="segmented">${localeButtons}</div></div></div><div class="hero-toolbar-group"><div class="hero-toolbar-label">${escapeHtml(t("aggregation"))}</div><div class="hero-toolbar-body"><div class="segmented">${aggButtons}</div>${aggregationInfoHtml()}</div></div><div class="hero-toolbar-group"><div class="hero-toolbar-label toolbar-inline-copy" title="${escapeHtml(t("tag_bar_help"))}">${escapeHtml(t("tag_bars"))}</div><div class="hero-toolbar-body"><div class="segmented">${tagButtons}</div></div></div>`;
}

function renderPass1(pass1) {
  if (!pass1) return "";
  const pass1View = aggregationPayload(pass1);
  if (!pass1View) return "";
  const sections = [];
  const globalTagCountMax = Object.values(pass1View.distributions || {}).flatMap((dist) => (
    Object.values(dist || {}).map((value) => Number(value) || 0)
  )).reduce((maxValue, value) => Math.max(maxValue, value), 0);

  const cards = [];
  const overview = pass1View.overview || {};
  cards.push({label: STATE.aggregationMode === "conversation" ? t("units_label") : t("samples"), value: pass1View.total || 0});
  cards.push({label: t("success"), value: `${(((overview.success_rate || 0) * 100).toFixed(1))}%`});
  cards.push({label: t("tokens"), value: Number(overview.total_tokens || 0).toLocaleString()});
  cards.push({label: t("arbitrated"), value: `${(((overview.arbitrated_rate || 0) * 100).toFixed(1))}%`});
  cards.push({label: "Prompt Mode", value: overview.prompt_mode || (overview.compact_prompt ? "compact" : "full")});
  if (overview.conversation_char_budget) cards.push({label: "Budget", value: Number(overview.conversation_char_budget || 0).toLocaleString()});
  if (overview.unmapped_unique) cards.push({label: t("unmapped"), value: overview.unmapped_unique});
  if (STATE.aggregationMode === "conversation" && (overview.llm_labeled_units || overview.inherited_units)) {
    cards.push({label: t("llm_labeled_units"), value: overview.llm_labeled_units || 0});
    cards.push({label: t("inherited_units"), value: overview.inherited_units || 0});
  } else if (overview.sparse_inherited) {
    cards.push({label: t("llm_labeled"), value: overview.sparse_labeled || 0});
    cards.push({label: t("inherited"), value: overview.sparse_inherited || 0});
  }
  const pass1Notes = [];
  if (STATE.aggregationMode === "conversation") {
    pass1Notes.push(`<div class="note">${escapeHtml(t("pass1_conversation_note"))}</div>`);
    if (DATA && DATA.title_key === "dashboard_title_scoring") {
      pass1Notes.push(`<div class="note">${escapeHtml(t("pass1_scoring_subset_note"))}</div>`);
    }
  }
  const pass1Intro = pass1Notes.join("");
  sections.push(section(t("labeling_overview"), `${pass1Intro}${renderCards(cards)}`, `${pass1View.total || 0} ${aggregationUnitLabel(pass1, "samples")}`, true));

  const unmappedDetails = pass1View.unmapped_details || {};
  const unmappedByDim = unmappedDetails.by_dimension || {};
  const unmappedPanels = Object.entries(unmappedByDim).map(([dim, rows]) => {
    const ranked = (rows || []).slice(0, 10).map((row) => ({
      label: row.label,
      count: row.count || 0,
      examples: Array.isArray(row.examples) ? row.examples : [],
    }));
    if (!ranked.length) return "";
    const htmlRows = ranked.map((row, index) => {
      const examples = row.examples.length
        ? row.examples.map((item) => `<div class="note">[${escapeHtml(item.id || "?")}] ${escapeHtml(item.query || "")}</div>`).join("")
        : `<div class="note">${escapeHtml(t("no_example_sample"))}</div>`;
      return `<tr>
        <td class="rank-cell">${index + 1}</td>
        <td class="rank-tag-cell"><span class="rank-tag-text" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</span></td>
        <td class="metric-cell">${fmtInt(row.count)}</td>
        <td>${examples}</td>
      </tr>`;
    }).join("");
    return `<div class="mini-panel"><h4>${escapeHtml(dimensionLabel(dim))}</h4>
      <table class="data-table compact-table">
        <thead><tr><th>#</th><th>${escapeHtml(t("tag"))}</th><th>${escapeHtml(t("count"))}</th><th>${escapeHtml(t("examples"))}</th></tr></thead>
        <tbody>${htmlRows}</tbody>
      </table>
    </div>`;
  }).join("");
  if (unmappedPanels) {
    const subtitle = t("top_10_per_dimension", { count: fmtInt(unmappedDetails.total_occurrences || 0) });
    sections.push(section(t("unmapped_tags"), `<div class="grid">${unmappedPanels}</div>`, subtitle, false));
  }

  const distPanels = Object.entries(pass1View.distributions || {}).map(([dim, dist]) => {
    const total = Object.values(dist || {}).reduce((sum, value) => sum + value, 0);
    const entries = Object.entries(dist || {})
      .sort((a, b) => (b[1] || 0) - (a[1] || 0) || a[0].localeCompare(b[0]))
      .map(([label, value]) => ({
        label,
        value,
        count: value,
      }));
    return `<div class="mini-panel"><h4>${escapeHtml(dimensionLabel(dim))}</h4>${renderRankingTable(entries, t("count"), (entry) => (
      total > 0 ? `${entry.value} (${((entry.value / total) * 100).toFixed(1)}%)` : `${entry.value}`
    ), (entry) => explorerPatchForTag(dim, entry.label, "quality_asc"), {
      barMetricKey: "count",
      barMetricLabel: t("count"),
      barColor: DIM_COLORS[dim] || "#2563eb",
      globalBarMax: globalTagCountMax,
    })}</div>`;
  }).join("");
  if (distPanels) {
    sections.push(section(t("tag_distributions"), `<div class="grid">${distPanels}</div>`, t("all_tags_sorted_frequency"), true));
  }

  const confStats = Object.entries(pass1View.confidence_stats || {}).map(([dim, stats]) => {
    return `<tr>
      <td>${escapeHtml(dimensionLabel(dim))}</td>
      <td><span style="display:inline-block;padding:4px 8px;border-radius:999px;background:${confColor(stats.mean)}">${fmt(stats.mean, 3)}</span></td>
      <td>${fmt(stats.min, 2)}</td>
      <td>${fmt(stats.max, 2)}</td>
      <td>${stats.below_threshold ?? 0}</td>
    </tr>`;
  }).join("");
  if (confStats) {
    sections.push(section(t("confidence"), `<table class="data-table"><thead><tr><th>${escapeHtml(t("dimension"))}</th><th>${escapeHtml(t("mean"))}</th><th>${escapeHtml(t("min"))}</th><th>${escapeHtml(t("max"))}</th><th>${escapeHtml(t("below_threshold"))}</th></tr></thead><tbody>${confStats}</tbody></table>`, "", false));
  }

  const cross = pass1View.cross_matrix || {};
  if ((cross.rows || []).length && (cross.cols || []).length) {
    let html = `<table class="data-table"><thead><tr><th>${escapeHtml(`${dimensionLabel("intent")} \\ ${dimensionLabel("difficulty")}`)}</th>${cross.cols.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}<th>${escapeHtml(t("total"))}</th></tr></thead><tbody>`;
    for (const row of cross.rows) {
      let total = 0;
      html += `<tr><td>${escapeHtml(row)}</td>`;
      for (const col of cross.cols) {
        const value = cross.data[`${row}|${col}`] || 0;
        total += value;
        html += `<td>${value || "-"}</td>`;
      }
      html += `<td>${total}</td></tr>`;
    }
    html += `</tbody></table>`;
    sections.push(section(t("intent_difficulty"), html, "", false));
  }

  const coverageItems = Object.entries(pass1View.coverage || {}).map(([dim, item]) => {
    if (!item.pool_size) return "";
    const pct = (item.rate || 0) * 100;
    const tags = item.unused && item.unused.length && item.unused.length <= 24
      ? `<div class="tags">${item.unused.map((tag) => `<button class="tag link-btn" type="button" data-explorer-patch="${escapeAttrJson(explorerPatchForTag(dim, tag, "quality_asc"))}">${escapeHtml(tag)}</button>`).join("")}</div>`
      : `<div class="note">${escapeHtml(t("unused_tags", { count: item.unused?.length || 0 }))}</div>`;
    return `<div class="mini-panel">
      <h4>${escapeHtml(dimensionLabel(dim))}</h4>
      <div class="note">${item.used}/${item.pool_size} (${pct.toFixed(1)}%)</div>
      ${renderBarChart([{label: dimensionLabel(dim), value: pct, display: `${pct.toFixed(1)}%`}], "#2563eb", 100)}
      ${tags}
    </div>`;
  }).join("");
  const extensionSection = renderExtensions(pass1.extensions);
  if (extensionSection) {
    sections.push(extensionSection);
  }

  if (coverageItems) {
    sections.push(section(t("pool_coverage"), `<div class="grid">${coverageItems}</div>`, "", false));
  }

  return sections.join("");
}

function formatCountList(items, limit = 6) {
  const entries = Object.entries(items || {}).filter(([, value]) => Number(value) > 0);
  if (!entries.length) return t("no_data");
  entries.sort((a, b) => (b[1] || 0) - (a[1] || 0) || a[0].localeCompare(b[0]));
  const shown = entries.slice(0, limit).map(([key, value]) => `${key} (${fmtInt(value)})`);
  const suffix = entries.length > limit ? ` +${entries.length - limit}` : "";
  return shown.join(", ") + suffix;
}

function renderExtensionInspectButton(label, patch, dataAttr = "data-extension-drilldown") {
  return `<button class="action-btn subtle-btn" type="button" ${dataAttr}="1" data-explorer-patch="${escapeAttrJson(patch)}">${escapeHtml(label)}</button>`;
}

function sortExtensionFieldEntries(ext) {
  const labels = ext.labels || {};
  const schema = ((ext.config || {}).schema) || {};
  return Object.entries(labels).sort((left, right) => {
    const leftTotal = Object.values(left[1] || {}).reduce((sum, value) => sum + Number(value || 0), 0);
    const rightTotal = Object.values(right[1] || {}).reduce((sum, value) => sum + Number(value || 0), 0);
    if (rightTotal !== leftTotal) return rightTotal - leftTotal;
    const leftOptions = Number((((schema[left[0]] || {}).option_count) || 0));
    const rightOptions = Number((((schema[right[0]] || {}).option_count) || 0));
    if (rightOptions !== leftOptions) return rightOptions - leftOptions;
    return left[0].localeCompare(right[0]);
  });
}

function renderExtensionConfig(ext) {
  const config = ext.config || {};
  const schema = config.schema || {};
  const schemaEntries = Object.entries(schema);
  const metaBits = [];
  if (config.description) metaBits.push(`<div class="note">${escapeHtml(config.description)}</div>`);
  if (config.source) metaBits.push(`<div class="note">${escapeHtml(t("extension_source"))}: ${escapeHtml(config.source)}</div>`);
  const promptBlock = config.prompt
    ? `<details class="drawer-collapse">
        <summary>${escapeHtml(t("extension_prompt"))}</summary>
        <div class="drawer-json extension-config-code">${escapeHtml(config.prompt)}</div>
      </details>`
    : "";
  const schemaBlock = schemaEntries.length
    ? `<details class="drawer-collapse">
        <summary>${escapeHtml(t("extension_schema"))}</summary>
        <div class="extension-schema-list">${schemaEntries.map(([fieldName, fieldConfig]) => {
          const info = [];
          if (fieldConfig.type) info.push(`${t("extension_field_type")}: ${fieldConfig.type}`);
          if (fieldConfig.option_count !== null && fieldConfig.option_count !== undefined) info.push(`${t("extension_option_count")}: ${fmtInt(fieldConfig.option_count)}`);
          const options = Array.isArray(fieldConfig.options) && fieldConfig.options.length
            ? `<div class="tags">${fieldConfig.options.map((option) => `<span class="tag">${escapeHtml(option)}</span>`).join("")}</div>`
            : "";
          return `<div class="extension-schema-item">
            <div class="extension-schema-header"><strong>${escapeHtml(fieldName)}</strong>${info.length ? `<span class="note"> · ${escapeHtml(info.join(" · "))}</span>` : ""}</div>
            ${fieldConfig.description ? `<div class="note">${escapeHtml(fieldConfig.description)}</div>` : ""}
            ${options}
          </div>`;
        }).join("")}</div>
      </details>`
    : "";
  if (!metaBits.length && !promptBlock && !schemaBlock) return "";
  return `<div class="extension-config">
    ${metaBits.join("")}
    ${promptBlock}
    ${schemaBlock}
  </div>`;
}

function renderExtensions(extensionsBucket) {
  const extView = aggregationPayload(extensionsBucket);
  if (!extView) return "";
  const extensions = extView.extensions || {};
  const entries = Object.entries(extensions).sort((left, right) => {
    const leftMatched = Number((left[1] || {}).matched || 0);
    const rightMatched = Number((right[1] || {}).matched || 0);
    if (rightMatched !== leftMatched) return rightMatched - leftMatched;
    const leftFieldCount = Number((((left[1] || {}).summary || {}).field_count) || 0);
    const rightFieldCount = Number((((right[1] || {}).summary || {}).field_count) || 0);
    if (rightFieldCount !== leftFieldCount) return rightFieldCount - leftFieldCount;
    return String(left[0]).localeCompare(String(right[0]));
  });
  if (!entries.length) return "";

  const panels = entries.map(([extId, ext]) => {
    const summaryInfo = ext.summary || {};
    const total = fmtInt(ext.total || 0);
    const matched = fmtInt(ext.matched || 0);
    const statusLine = formatCountList(ext.status_counts);
    const versionLine = formatCountList(ext.spec_versions);
    const hashLine = formatCountList(ext.spec_hashes);
    const summary = [
      `${escapeHtml(t("total"))}: ${total}`,
      `${escapeHtml(t("extension_matched"))}: ${matched}`,
      `${escapeHtml(t("extension_status"))}: ${escapeHtml(statusLine)}`,
    ];
    if (versionLine !== t("no_data")) summary.push(`${escapeHtml(t("extension_versions"))}: ${escapeHtml(versionLine)}`);
    if (hashLine !== t("no_data")) summary.push(`${escapeHtml(t("extension_hashes"))}: ${escapeHtml(hashLine)}`);

    const summaryCards = renderCards([
      {label: t("extension_matched"), value: `${matched} / ${total}`},
      {label: t("matched_rate"), value: `${((Number(summaryInfo.matched_rate) || 0) * 100).toFixed(1)}%`},
      {label: t("extension_fields_used"), value: fmtInt(summaryInfo.field_count || 0)},
      {label: t("extension_values"), value: fmtInt(summaryInfo.value_count || 0)},
      {label: t("total"), value: total},
    ]);

    const fieldEntries = sortExtensionFieldEntries(ext);
    const labelPanels = fieldEntries.map(([field, dist]) => {
      const fieldConfig = (((ext.config || {}).schema) || {})[field] || {};
      const totalCount = Object.values(dist || {}).reduce((sum, value) => sum + value, 0);
      const entries = Object.entries(dist || {}).map(([label, value]) => ({
        label,
        value,
        count: value,
        extensionTag: `ext:${extId}:${field}:${label}`,
      }));
      if (!entries.length) return "";
      const maxValue = Math.max(...entries.map((entry) => Number(entry.value) || 0), 0);
      const fieldMeta = [];
      if (fieldConfig.type) fieldMeta.push(`${t("extension_field_type")}: ${fieldConfig.type}`);
      if (fieldConfig.option_count !== null && fieldConfig.option_count !== undefined) fieldMeta.push(`${t("extension_option_count")}: ${fmtInt(fieldConfig.option_count)}`);
      return `<div class="mini-panel">
        <div class="extension-field-header">
          <div>
            <h4>${escapeHtml(field)}</h4>
            ${fieldMeta.length ? `<div class="note">${escapeHtml(fieldMeta.join(" · "))}</div>` : ""}
            ${fieldConfig.description ? `<div class="note">${escapeHtml(fieldConfig.description)}</div>` : ""}
          </div>
          ${renderExtensionInspectButton(
            t("inspect_field_samples"),
            { tagQueryAppend: [`extfield:${extId}:${field}`], sort: "sample_id_asc" },
          )}
        </div>
        ${renderRankingTable(entries, t("count"), (entry) => (
          totalCount > 0 ? `${entry.value} (${((entry.value / totalCount) * 100).toFixed(1)}%)` : `${entry.value}`
        ), (entry) => ({
          tagQueryAppend: [entry.extensionTag],
          sort: "sample_id_asc",
        }), {
          barMetricKey: "count",
          barMetricLabel: t("count"),
          barColor: "#0ea5e9",
          globalBarMax: maxValue,
          buttonDataAttr: "data-extension-drilldown",
        })}
      </div>`;
    }).join("");

    let labelsHtml = labelPanels;
    if (!labelPanels) {
      labelsHtml = `<div class="note">${escapeHtml(t("no_tags"))}</div>`;
    } else {
      labelsHtml = `<div class="grid">${labelPanels}</div>`;
    }

    const unmappedDetails = ext.unmapped_details || {};
    const unmappedRows = Object.entries(unmappedDetails.by_dimension || {}).flatMap(([dim, rows]) => (
      (rows || []).slice(0, 6).map((row) => `<div class="note">${escapeHtml(dim)}: ${escapeHtml(row.label)} (${fmtInt(row.count || 0)})</div>`)
    ));
    const unmappedHtml = unmappedRows.length
      ? `<div class="note">${escapeHtml(t("unmapped"))}:<br>${unmappedRows.join("")}</div>`
      : "";
    const configHtml = renderExtensionConfig(ext);
    const inspectMatched = renderExtensionInspectButton(
      t("inspect_matched_samples"),
      { tagQueryAppend: [`extid:${extId}`], sort: "sample_id_asc" },
    );
    const title = ((ext.config || {}).display_name) || extId;

    return `<div class="mini-panel">
      <div class="extension-panel-header">
        <div>
          <h4>${escapeHtml(title)}</h4>
          ${title !== extId ? `<div class="note">${escapeHtml(extId)}</div>` : ""}
        </div>
        <div class="extension-action-row">${inspectMatched}</div>
      </div>
      ${summaryCards}
      <div class="note">${summary.join("<br>")}</div>
      ${configHtml ? `<div class="drawer-section"><h4>${escapeHtml(t("extension_configuration"))}</h4>${configHtml}</div>` : ""}
      <div class="drawer-section"><h4>${escapeHtml(t("extension_details"))}</h4>${labelsHtml}</div>
      ${unmappedHtml}
    </div>`;
  }).join("");

  return section(t("extension_labels"), `<div class="grid">${panels}</div>`, t("extension_fields"), true);
}

function renderPass2(pass2) {
  if (!pass2) return "";
  const pass2View = aggregationPayload(pass2);
  if (!pass2View) return "";
  const sections = [];
  const overview = pass2View.overview || {};
  const cards = [
    {label: t("scored"), value: overview.total_scored || 0},
    {label: t("failed"), value: overview.total_failed || 0},
    {label: t("mean_value"), value: fmt(overview.mean_value, 1), className: scoreClass(overview.mean_value)},
    ...(overview.mean_value_v2 ? [{label: t("value_score_v2"), value: fmt(overview.mean_value_v2, 1), className: scoreClass(overview.mean_value_v2)}] : []),
    ...(overview.mean_selection_v2 ? [{label: t("selection_v2"), value: fmt(overview.mean_selection_v2, 1), className: scoreClass(overview.mean_selection_v2)}] : []),
    {label: t("complexity"), value: fmt(overview.mean_complexity, 1), className: scoreClass(overview.mean_complexity)},
    {label: t("quality"), value: fmt(overview.mean_quality, 1), className: scoreClass(overview.mean_quality)},
    {label: t("median_rarity"), value: fmt(overview.median_rarity, 1), className: scoreClass(overview.median_rarity)},
    ...(overview.mean_extension_rarity ? [{label: t("extension_rarity_preview"), value: fmt(overview.mean_extension_rarity, 1), className: scoreClass(overview.mean_extension_rarity)}] : []),
    ...(overview.mean_rarity_v2 ? [{label: t("rarity_v2"), value: fmt(overview.mean_rarity_v2, 1), className: scoreClass(overview.mean_rarity_v2)}] : []),
    {label: t("confidence"), value: fmt(overview.mean_confidence, 2)},
    {label: t("tokens"), value: Number(overview.total_tokens || 0).toLocaleString()},
    {label: "Prompt Mode", value: overview.prompt_mode || (overview.compact_prompt ? "compact" : "full")},
    ...(overview.extension_rarity_mode ? [{label: t("extension_rarity_mode"), value: overview.extension_rarity_mode}] : []),
    ...(overview.extension_baseline_source ? [{label: t("extension_baseline_source"), value: overview.extension_baseline_source}] : []),
  ];
  if (overview.value_truncation_budget) cards.push({label: "Budget", value: Number(overview.value_truncation_budget || 0).toLocaleString()});
  const pass2Intro = STATE.aggregationMode === "conversation"
    ? `<div class="note">${escapeHtml(t("pass2_conversation_note"))}</div>`
    : "";
  sections.push(section(t("scoring_overview"), `${pass2Intro}${renderCards(cards)}`, `${overview.total_scored || 0} ${aggregationUnitLabel(pass2, "samples")}`, true));

  const histograms = pass2View.histograms || {};
  const scoreDistributions = pass2View.score_distributions || {};
  const histogramBlocks = Object.entries(histograms).map(([key, bins]) => {
    if (!Array.isArray(bins) || !bins.length) return "";
    let bucketAction = null;
    if (key === "value_score") bucketAction = (bucket) => ({ maxValue: String(bucket), sort: "value_asc" });
    if (key === "quality_overall") bucketAction = (bucket) => ({ maxQuality: String(bucket), sort: "quality_asc" });
    if (key === "selection_score") bucketAction = (bucket) => ({ minSelection: String(bucket), sort: "selection_desc" });
    if (key === "confidence") bucketAction = (bucket) => ({ maxConfidence: String((bucket / 10).toFixed(1)), sort: "confidence_asc" });
    return renderHistogram(t(HIST_LABELS[key] || key), bins, HIST_COLORS[key] || "#2563eb", scoreDistributions[key], bucketAction);
  }).join("");
  if (histogramBlocks) {
    sections.push(section(t("score_distributions"), `<div class="grid">${histogramBlocks}</div>`, "", true));
  }

  if ((pass2View.confidence_histogram || []).some((value) => value > 0)) {
    const bins = pass2View.confidence_histogram;
    const max = Math.max(...bins, 1);
    const bars = bins.map((count, idx) => {
      const height = Math.max((count / max) * 100, 2);
      return `<div class="hist-bar" style="height:${height}%;background:#7c3aed" title="${(idx / 10).toFixed(1)}-${((idx + 1) / 10).toFixed(1)}: ${count}"></div>`;
    }).join("");
    const labels = bins.map((_, idx) => `<span>${(idx / 10).toFixed(1)}</span>`).join("");
    sections.push(section(t("confidence_distribution"), `<div class="mini-panel"><h4>${escapeHtml(t("llm_confidence"))}</h4><div class="hist-row">${bars}</div><div class="hist-labels">${labels}</div></div>`, "", false));
  }

  const byTagSections = [];
  for (const [title, bucket] of [
    [t("value_by_tag"), pass2View.value_by_tag || {}],
    [t("selection_by_tag"), pass2View.selection_by_tag || {}],
  ]) {
    const globalTagCountMax = Object.values(bucket).flatMap((dist) => (
      Object.values(dist || {}).map((info) => Number(info?.n) || 0)
    )).reduce((maxValue, value) => Math.max(maxValue, value), 0);
    const panels = Object.entries(bucket).map(([dim, dist]) => {
      const items = Object.entries(dist || {})
        .sort((a, b) => (b[1]?.mean || 0) - (a[1]?.mean || 0) || a[0].localeCompare(b[0]))
        .map(([label, info]) => ({
          label,
          mean: info.mean,
          n: info.n || 0,
        }));
      const metricLabel = title === t("selection_by_tag") ? t("selection") : t("value");
      return `<div class="mini-panel"><h4>${escapeHtml(dimensionLabel(dim))}</h4>${renderRankingTable(items, metricLabel, (item) => `${fmt(item.mean, 2)} (n=${item.n})`, (item) => explorerPatchForTag(dim, item.label, title === t("selection_by_tag") ? "selection_desc" : "quality_asc"), {
        barMetricKey: "n",
        barMetricLabel: t("count"),
        barColor: DIM_COLORS[dim] || "#2563eb",
        globalBarMax: globalTagCountMax,
      })}</div>`;
    }).join("");
    if (panels) {
      byTagSections.push(section(title, `<div class="grid">${panels}</div>`, t("all_tags_sorted_score"), false));
    }
  }
  sections.push(byTagSections.join(""));

  const thinkingMode = pass2View.thinking_mode_stats || {};
  const flags = pass2View.flag_counts || {};
  if (Object.keys(thinkingMode).length || Object.keys(flags).length) {
    let html = `<div class="grid">`;
    if (Object.keys(thinkingMode).length) {
      const rows = Object.entries(thinkingMode).map(([mode, stats]) => `<tr>
        <td>${escapeHtml(thinkingModeLabel(mode))}</td>
        <td>${stats.count || 0}</td>
        <td>${fmt(stats.mean_value, 1)}</td>
        <td>${fmt(stats.mean_quality, 1)}</td>
        <td>${fmt(stats.mean_reasoning, 1)}</td>
      </tr>`).join("");
      html += `<div class="mini-panel"><h4>${escapeHtml(t("thinking_mode"))}</h4><table class="data-table"><thead><tr><th>${escapeHtml(t("mode"))}</th><th>${escapeHtml(t("count"))}</th><th>${escapeHtml(t("value"))}</th><th>${escapeHtml(t("quality"))}</th><th>${escapeHtml(t("reasoning"))}</th></tr></thead><tbody>${rows}</tbody></table></div>`;
    }
    if (Object.keys(flags).length) {
      const items = Object.entries(flags).slice(0, 18).map(([label, value]) => ({
        label,
        value,
        display: `${value}`,
      }));
      html += `<div class="mini-panel"><h4>${escapeHtml(t("flags"))}</h4>${renderBarChart(items, "#dc2626", null, (item) => ({ flagQuery: item.label, hasFlags: true, sort: "quality_asc" }))}</div>`;
    }
    html += `</div>`;
    sections.push(section(t("analysis"), html, "", true));
  }

  const perFile = pass2View.per_file_summary || [];
  if (perFile.length > 1) {
    const sortColumn = FILE_RANKING_COLUMNS[STATE.fileRankingSort.key] || FILE_RANKING_COLUMNS.mean_value;
    const keepRateButtons = KEEP_RATE_THRESHOLDS.map((threshold) => (
      `<button class="segmented-btn ${STATE.keepRateThreshold === threshold ? "active" : ""}" type="button" data-keep-rate-threshold="${escapeHtml(threshold)}">≥${escapeHtml(Number(threshold).toFixed(0))}</button>`
    )).join("");
    const rows = sortPerFileSummary(perFile)
      .map((row) => `<tr>
        <td><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ sourcePath: row.file, sort: "quality_asc" })}">${escapeHtml(row.file)}</button></td>
        <td>${row.count || 0}</td>
        <td><span class="${scoreClass(row.mean_value)}">${fmt(row.mean_value, 1)}</span></td>
        <td>${fmt(row.mean_complexity, 1)}</td>
        <td>${fmt(row.mean_quality, 1)}</td>
        <td>${fmt(row.mean_rarity, 1)}</td>
        <td>${fmt(row.mean_selection, 1)}</td>
        <td><span class="keep-rate-chip ${keepRateClass(fileKeepRateValue(row))}" title="${escapeHtml(`${keepRateLabel()} · selection ≥ ${Number(STATE.keepRateThreshold).toFixed(0)}`)}">${fmt(fileKeepRateValue(row) * 100, 0)}%</span></td>
        <td>${fmt(row.mean_turns, 1)}</td>
      </tr>`).join("");
    sections.push(section(
      t("file_ranking"),
      `<div class="file-ranking-controls"><div class="toolbar-inline-copy">${escapeHtml(t("keep_rate_threshold"))}</div><div class="segmented">${keepRateButtons}</div></div><table class="data-table"><thead><tr>${
        ["file", "count", "mean_value", "mean_complexity", "mean_quality", "mean_rarity", "mean_selection", "keep_rate_dynamic", "mean_turns"]
          .map(renderFileRankingHeader)
          .join("")
      }</tr></thead><tbody>${rows}</tbody></table>`,
      t("sorted_by", { count: perFile.length, column: STATE.fileRankingSort.key === "keep_rate_dynamic" ? keepRateLabel() : t(sortColumn.labelKey) }),
      true,
    ));
  }

  const weights = pass2View.weights_used || {};
  if (Object.keys(weights).length) {
    const formula = Object.entries(weights).map(([key, value]) => `${value}×${key}`).join(" + ");
    sections.push(section(t("configuration"), `<div class="note">${escapeHtml(formula)}</div>`, "", false));
  }

  const thresholds = pass2View.selection_thresholds || {};
  if (Object.keys(thresholds).length) {
    const rows = Object.entries(thresholds).map(([label, info]) => `<tr>
      <td>${escapeHtml(label)}</td>
      <td>${fmt(info.threshold, 1)}</td>
      <td>${info.count || 0}</td>
      <td><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ minSelection: String(info.threshold), sort: "selection_desc" })}">${escapeHtml(t("inspect_samples"))}</button></td>
    </tr>`).join("");
    sections.push(section(t("selection_thresholds"), `<table class="data-table"><thead><tr><th>${escapeHtml(t("band"))}</th><th>${escapeHtml(t("threshold"))}</th><th>${escapeHtml(t("count"))}</th><th>${escapeHtml(t("action"))}</th></tr></thead><tbody>${rows}</tbody></table>`, "", false));
  }

  const coverage = pass2View.coverage_at_thresholds || {};
  if (Object.keys(coverage).length) {
    const rows = Object.entries(coverage).map(([threshold, info]) => `<tr>
      <td>${escapeHtml(threshold)}</td>
      <td>${info.retained || 0}</td>
      <td>${fmt((info.pct || 0) * 100, 1)}%</td>
      <td>${fmt((info.coverage || 0) * 100, 1)}%</td>
      <td><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ minValue: threshold, sort: "value_asc" })}">${escapeHtml(t("inspect_retained"))}</button></td>
    </tr>`).join("");
    sections.push(section(t("coverage_at_thresholds"), `<table class="data-table"><thead><tr><th>${escapeHtml(t("value_min"))}</th><th>${escapeHtml(t("retained"))}</th><th>${escapeHtml(t("sample_pct"))}</th><th>${escapeHtml(t("tag_coverage"))}</th><th>${escapeHtml(t("action"))}</th></tr></thead><tbody>${rows}</tbody></table>`, "", false));
  }

  const flagImpact = pass2View.flag_value_impact || {};
  if (Object.keys(flagImpact).length) {
    const rows = Object.entries(flagImpact).sort((a, b) => (a[1].mean_value || 0) - (b[1].mean_value || 0)).map(([flag, info]) => `<tr>
      <td><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ flagQuery: flag, hasFlags: true, sort: "quality_asc" })}">${escapeHtml(flag)}</button></td>
      <td>${fmt(info.mean_value, 2)}</td>
      <td>${info.count || 0}</td>
    </tr>`).join("");
    sections.push(section(t("flag_impact"), `<table class="data-table"><thead><tr><th>${escapeHtml(t("flag"))}</th><th>${escapeHtml(t("mean_value"))}</th><th>${escapeHtml(t("count"))}</th></tr></thead><tbody>${rows}</tbody></table>`, "", false));
  }

  return sections.join("");
}

function renderConversations(conversation) {
  if (!conversation) return "";
  const cards = [
    {label: t("conversations"), value: conversation.total || 0},
    {label: t("conv_value"), value: fmt(conversation.mean_conv_value, 1), className: scoreClass(conversation.mean_conv_value)},
    {label: t("conv_selection"), value: fmt(conversation.mean_conv_selection, 1), className: scoreClass(conversation.mean_conv_selection)},
    {label: t("peak_complexity"), value: fmt(conversation.mean_peak_complexity, 1), className: scoreClass(conversation.mean_peak_complexity)},
    {label: t("mean_turns"), value: fmt(conversation.mean_turns, 1)},
    {label: "Compression Gap", value: fmt(conversation.mean_peak_minus_mean, 2)},
    {label: "Late-turn Gain", value: fmt(conversation.mean_late_turn_gain, 2)},
    {label: "Tool Turn Ratio", value: fmt((conversation.mean_tool_turn_ratio || 0) * 100, 0) + "%"},
    {label: "Unique Tools", value: fmt(conversation.mean_unique_tool_count, 1)},
    {label: "Unique Files", value: fmt(conversation.mean_unique_file_count, 1)},
    {label: t("observed_ratio"), value: fmt((conversation.mean_observed_turn_ratio || 0) * 100, 0) + "%", sub: t("convs_lt_50_observed", { count: fmtInt(conversation.low_observed_coverage_count || 0) })},
    {label: t("rarity_confidence"), value: fmt((conversation.mean_rarity_confidence || 0), 2), sub: t("convs_lt_060_rarity_conf", { count: fmtInt(conversation.low_rarity_confidence_count || 0) })},
  ];

  let body = `<div class="note">${escapeHtml(t("conversation_panel_note"))}</div>${renderCards(cards)}`;
  const histBlocks = [];
  for (const [label, bins, color] of [
    [t("conv_value"), conversation.conv_value_hist, "#2563eb"],
    [t("conv_selection"), conversation.conv_selection_hist, "#be185d"],
    [t("peak_complexity"), conversation.peak_complexity_hist, "#c2410c"],
  ]) {
    if (Array.isArray(bins) && bins.some((value) => value > 0)) {
      let bucketAction = null;
      if (label === t("conv_value")) bucketAction = (bucket) => ({ convValueMin: String(bucket), sort: "conv_value_desc" });
      if (label === t("conv_selection")) bucketAction = (bucket) => ({ convSelectionMin: String(bucket), sort: "conv_selection_desc" });
      if (label === t("peak_complexity")) bucketAction = (bucket) => ({ peakComplexityMin: String(bucket), sort: "conv_value_desc" });
      histBlocks.push(renderHistogram(label, bins, color, null, bucketAction));
    }
  }
  const turnDist = conversation.turn_distribution || {};
  if (Object.keys(turnDist).length) {
    const items = Object.entries(turnDist).map(([turns, count]) => ({
      label: t("turns_label", { count: turns }),
      value: count,
      display: `${count}`,
      turns: Number(turns),
    }));
    histBlocks.push(`<div class="mini-panel"><h4>${escapeHtml(t("turn_distribution"))}</h4>${renderBarChart(items, "#0f766e", null, (item) => ({ turnCountMin: String(item.turns), sort: "turn_count_desc" }))}</div>`);
  }
  const observedBands = conversation.observed_turn_ratio_bands || {};
  if (Object.keys(observedBands).length) {
    const items = Object.entries(observedBands).map(([label, count], index) => ({
      label,
      value: count,
      display: `${count}`,
      threshold: [0.0, 0.25, 0.5, 0.75][index] ?? 0.0,
    }));
    histBlocks.push(`<div class="mini-panel"><h4>${escapeHtml(t("observed_turn_coverage"))}</h4>${renderBarChart(items, "#0f766e", null, (item) => ({ observedTurnRatioMin: String(item.threshold), sort: "observed_turn_ratio_desc" }))}</div>`);
  }
  const rarityBands = conversation.rarity_confidence_bands || {};
  if (Object.keys(rarityBands).length) {
    const items = Object.entries(rarityBands).map(([label, count], index) => ({
      label,
      value: count,
      display: `${count}`,
      threshold: [0.0, 0.25, 0.5, 0.75][index] ?? 0.0,
    }));
    histBlocks.push(`<div class="mini-panel"><h4>${escapeHtml(t("rarity_confidence"))}</h4>${renderBarChart(items, "#7c3aed", null, (item) => ({ rarityConfidenceMin: String(item.threshold), sort: "rarity_confidence_desc" }))}</div>`);
  }
  if (histBlocks.length) {
    body += `<div class="grid" style="margin-top:14px">${histBlocks.join("")}</div>`;
  }
  return section(t("conversation_aggregation"), body, "", false);
}

function explorerSortOptions(includeScores = true) {
  const options = [
    ["sample_id_asc", t("sample_id")],
  ];
  if (includeScores) {
    options.unshift(
      ["quality_asc", t("quality_asc")],
      ["value_asc", t("value_asc")],
      ["confidence_asc", t("confidence_asc")],
      ["selection_desc", t("selection_desc")],
      ["conv_value_desc", t("conv_value_desc")],
      ["conv_selection_desc", t("conv_selection_desc")],
      ["turn_count_desc", t("turn_count_desc")],
      ["observed_turn_ratio_asc", t("observed_ratio_asc")],
      ["observed_turn_ratio_desc", t("observed_ratio_desc")],
      ["rarity_confidence_asc", t("rarity_confidence_asc")],
      ["rarity_confidence_desc", t("rarity_confidence_desc")],
      ["rarity_desc", t("rarity_desc")],
    );
  }
  return options;
}

function explorerQueryHasFilters(query) {
  if (!query) return false;
  return [
    query.tagQuery,
    query.language,
    query.intent,
    query.sourcePath,
    query.textQuery,
    query.flagQuery,
    query.convValueMin,
    query.convSelectionMin,
    query.convSelectionMax,
    query.peakComplexityMin,
    query.turnCountMin,
    query.observedTurnRatioMin,
    query.observedTurnRatioMax,
    query.rarityConfidenceMin,
    query.rarityConfidenceMax,
    query.minValue,
    query.maxValue,
    query.maxQuality,
    query.minSelection,
    query.maxSelection,
    query.maxConfidence,
    query.thinkingMode,
  ].some((value) => String(value || "").trim() !== "")
    || !!query.hasFlags
    || !!query.includeInherited;
}

function renderExplorerActiveFilters(query) {
  const tagTokens = splitCsv((query || {}).tagQuery);
  if (!tagTokens.length) return "";
  const chips = tagTokens.map((token) => {
    const isExtension = !!parseExtensionQueryToken(token);
    const className = isExtension ? "tag explorer-filter-chip extension-tag" : "tag explorer-filter-chip";
    return `<button class="${className}" type="button" data-explorer-remove-tag="${escapeHtml(token)}" title="${escapeHtml(t("remove_filter"))}">
      <span>${escapeHtml(formatExplorerQueryToken(token))}</span>
      <span class="chip-remove" aria-hidden="true">×</span>
    </button>`;
  }).join("");
  return `<div class="explorer-active-filters">
    <div class="note">${escapeHtml(t("active_filters"))}</div>
    <div class="tags">${chips}</div>
  </div>`;
}

function renderExplorerResults(rows) {
  if (!rows.length) {
    return `<div class="empty">${escapeHtml(t("run_query_or_choose_preset"))}</div>`;
  }
  const body = rows.map((row) => {
    const coreTags = (row.flat_tags || []).slice(0, 4).map((tag) => `<span class="preview-tag">${escapeHtml(tag)}</span>`).join("");
    const extensionTags = (row.extension_tags || [])
      .filter((tag) => String(tag || "").startsWith("ext:"))
      .slice(0, 3)
      .map((tag) => `<span class="preview-tag extension-tag" title="${escapeHtml(tag)}">${escapeHtml(formatExplorerQueryToken(tag))}</span>`)
      .join("");
    const tags = [coreTags, extensionTags].filter(Boolean).join("");
    const sampleCell = row.detail_chunk
      ? `<button class="link-btn" data-explorer-detail="${escapeHtml(row.doc_id)}" data-explorer-scope="${escapeHtml(row.scope_id)}" data-explorer-detail-chunk="${escapeHtml(row.detail_chunk)}">${escapeHtml(row.sample_id || row.doc_id)}</button>`
      : escapeHtml(row.sample_id || row.doc_id);
    return `<tr>
      <td>${sampleCell}</td>
      <td class="preview-query" title="${escapeHtml(row.query_preview || "")}">${textPreview(row.query_preview)}</td>
      <td class="preview-response" title="${escapeHtml(row.response_preview || "")}">${textPreview(row.response_preview)}</td>
      <td><div class="preview-tags">${tags || `<span class="note">-</span>`}</div></td>
      <td>${row.value_score === null || row.value_score === undefined ? "-" : `<span class="${scoreClass(row.value_score)}">${fmt(row.value_score, 1)}</span>`}</td>
      <td>${row.quality_overall === null || row.quality_overall === undefined ? "-" : `<span class="${scoreClass(row.quality_overall)}">${fmt(row.quality_overall, 1)}</span>`}</td>
      <td>${row.selection_score === null || row.selection_score === undefined ? "-" : fmt(row.selection_score, 1)}</td>
      <td>${row.conv_value === null || row.conv_value === undefined ? "-" : `<span class="${scoreClass(row.conv_value)}">${fmt(row.conv_value, 1)}</span>`}</td>
      <td>${row.observed_turn_ratio === null || row.observed_turn_ratio === undefined ? "-" : fmt(row.observed_turn_ratio, 2)}</td>
      <td>${row.rarity_confidence === null || row.rarity_confidence === undefined ? "-" : fmt(row.rarity_confidence, 2)}</td>
      <td>${row.turn_count === null || row.turn_count === undefined ? "-" : fmtInt(row.turn_count)}</td>
      <td title="${escapeHtml(row.source_file || row.scope_path || "")}"><button class="link-btn" type="button" data-explorer-patch="${escapeAttrJson({ sourcePath: row.scope_path || row.source_file || "", sort: "quality_asc" })}">${escapeHtml(row.scope_path || row.source_file || "-")}</button></td>
    </tr>`;
  }).join("");
  return `<div class="result-table-wrap"><table class="data-table">
    <thead><tr><th>${escapeHtml(t("sample"))}</th><th>${escapeHtml(t("query"))}</th><th>${escapeHtml(t("response"))}</th><th>${escapeHtml(t("tags"))}</th><th>${escapeHtml(t("value"))}</th><th>${escapeHtml(t("quality"))}</th><th>${escapeHtml(t("selection"))}</th><th>${escapeHtml(t("conv_value"))}</th><th>${escapeHtml(t("observed"))}</th><th>${escapeHtml(t("rarity_confidence"))}</th><th>${escapeHtml(t("turns"))}</th><th>${escapeHtml(t("source"))}</th></tr></thead>
    <tbody>${body}</tbody>
  </table></div>`;
}

function renderDrawerExtensionPayloads(payload) {
  const extensions = payload || {};
  const entries = Object.entries(extensions);
  if (!entries.length) return `<span class="note">${escapeHtml(t("no_tags"))}</span>`;
  return `<div class="grid">${entries.map(([extId, extPayload]) => {
    const labels = extPayload.labels || {};
    const confidence = extPayload.confidence || {};
    const labelRows = Object.entries(labels).map(([field, value]) => {
      const rendered = Array.isArray(value)
        ? value.map((item) => `<span class="tag extension-tag">${escapeHtml(item)}</span>`).join("")
        : value ? `<span class="tag extension-tag">${escapeHtml(value)}</span>` : `<span class="note">-</span>`;
      const conf = confidence[field];
      return `<div class="extension-schema-item">
        <div class="extension-schema-header"><strong>${escapeHtml(field)}</strong>${conf !== null && conf !== undefined ? `<span class="note"> · ${escapeHtml(t("confidence"))}: ${escapeHtml(fmt(conf, 2))}</span>` : ""}</div>
        <div class="tags">${rendered}</div>
      </div>`;
    }).join("");
    const unmapped = (extPayload.unmapped || []).map((item) => `<span class="tag">${escapeHtml(`${item.dimension}:${item.value}`)}</span>`).join("");
    return `<div class="mini-panel">
      <h4>${escapeHtml(extId)}</h4>
      <div class="note">${escapeHtml(t("extension_status"))}: ${escapeHtml(extPayload.status || "unknown")} · ${escapeHtml(t("extension_matched"))}: ${escapeHtml(String(!!extPayload.matched))}</div>
      ${labelRows || `<div class="note">${escapeHtml(t("no_tags"))}</div>`}
      ${unmapped ? `<div class="note">${escapeHtml(t("unmapped"))}</div><div class="tags">${unmapped}</div>` : ""}
    </div>`;
  }).join("")}</div>`;
}

function renderExplorerDrawer() {
  if (!STATE.explorer.detailDocId) return "";
  const detail = STATE.explorer.detailData;
  const loading = STATE.explorer.detailLoading;
  const header = loading
    ? `<div class="drawer-title">${escapeHtml(t("loading_sample"))}</div>`
    : `<div><div class="drawer-title">${escapeHtml((detail && detail.id) || STATE.explorer.detailDocId)}</div>
        <div class="drawer-sub">${escapeHtml((((detail || {}).metadata || {}).source_file) || STATE.explorer.detailScopeId || "")}</div></div>`;

  let body = `<div class="drawer-section"><div class="note">${loading ? escapeHtml(t("loading_full_sample_payload")) : escapeHtml((DATA.explorer || {}).detail_limit_notice || "")}</div></div>`;
  if (detail && !loading) {
    const labels = detail.labels || {};
    const extensionPayloads = detail.label_extensions || {};
    const value = detail.value || {};
    const flatTags = Object.entries(labels).flatMap(([key, val]) => {
      if (["confidence", "unmapped", "inherited", "inherited_from"].includes(key)) return [];
      if (["confidence", "canonicalized", "unmapped", "inherited", "inherited_from"].includes(key)) return [];
      const label = dimensionLabel(key);
      if (Array.isArray(val)) return val.map((item) => `${label}:${item}`);
      return val ? [`${label}:${val}`] : [];
    });
    const cards = [
      [t("value"), value.value_score],
      [t("quality"), (value.quality || {}).overall],
      [t("selection"), value.selection_score],
      ...(value.value_score_v2 !== null && value.value_score_v2 !== undefined ? [[t("value_score_v2"), value.value_score_v2]] : []),
      ...(value.selection_score_v2 !== null && value.selection_score_v2 !== undefined ? [[t("selection_v2"), value.selection_score_v2]] : []),
      ...(((value.rarity_extension || {}).score) !== null && ((value.rarity_extension || {}).score) !== undefined
        ? [[t("extension_rarity_preview"), ((value.rarity_extension || {}).score)]]
        : []),
      ...(((value.rarity_v2 || {}).score) !== null && ((value.rarity_v2 || {}).score) !== undefined
        ? [[t("rarity_v2"), ((value.rarity_v2 || {}).score)]]
        : []),
      [t("conv_value"), (detail.conversation || {}).conv_value],
      [t("conv_sel"), (detail.conversation || {}).conv_selection],
      [t("conv_rarity"), (detail.conversation || {}).conv_rarity],
      [t("observed"), (detail.conversation || {}).observed_turn_ratio],
      [t("rarity_confidence"), (detail.conversation || {}).rarity_confidence],
      [t("confidence"), value.confidence],
      [t("thinking"), value.thinking_mode || ((detail.metadata || {}).thinking_mode || "")],
      [t("turns"), (detail.conversation || {}).turn_count || (detail.metadata || {}).total_turns || (detail.conversations || []).length],
      [t("peak_cplx"), (detail.conversation || {}).peak_complexity],
      ["Top-5 Mean", (((detail.conversation || {}).detail || {}).top_k_mean)],
      ["Bottom-3 Mean", (((detail.conversation || {}).detail || {}).bottom_k_mean)],
      ["Value Std", (((detail.conversation || {}).detail || {}).turn_value_std)],
      ["Late Gain", (detail.conversation || {}).late_turn_gain],
      ["Tool Ratio", (detail.conversation || {}).tool_turn_ratio],
      ["Unique Tools", (detail.conversation || {}).unique_tool_count],
      ["Unique Files", (detail.conversation || {}).unique_file_count],
    ].map(([label, raw]) => `<div class="kv"><div class="kv-label">${escapeHtml(label)}</div><div class="kv-value">${escapeHtml(raw === null || raw === undefined || raw === "" ? "-" : (typeof raw === "number" ? fmt(raw, label === t("confidence") ? 2 : 1) : raw))}</div></div>`).join("");
    body = `
      <div class="drawer-section"><div class="drawer-grid">${cards}</div></div>
      <div class="drawer-section"><h4>${escapeHtml(t("tags"))}</h4><div class="tags">${flatTags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("") || `<span class="note">${escapeHtml(t("no_tags"))}</span>`}</div></div>
      <div class="drawer-section"><h4>${escapeHtml(t("extension_labels"))}</h4>${renderDrawerExtensionPayloads(extensionPayloads)}</div>
      <div class="drawer-section"><h4>${escapeHtml(t("conversation"))}</h4>${renderDrawerTurns(detail.conversations || [])}</div>
      <div class="drawer-section"><h4>${escapeHtml(t("json"))}</h4>${renderDrawerJson(detail)}</div>
    `;
  }

  return `<div class="drawer-backdrop" data-explorer-close="1"></div>
    <aside class="drawer">
      <div class="drawer-header">
        ${header}
        <button class="drawer-close" type="button" data-explorer-close="1">×</button>
      </div>
      <div class="drawer-body">${body}</div>
    </aside>`;
}

function renderExplorer(scope) {
  if (!scopeSupportsExplorer(scope)) return "";
  const totalSamples = explorerTotalSamples(scope);
  const query = STATE.explorer.queryModel;
  const supportsScores = scopeHasPass2Data(scope);
  const candidateFileCount = explorerFileScopeIds(scope, query).length;
  const availableSorts = explorerSortOptions(supportsScores);
  if (!availableSorts.some(([value]) => value === STATE.explorer.sort)) {
    STATE.explorer.sort = "sample_id_asc";
  }
  const sortOptions = availableSorts.map(([value, label]) => (
    `<option value="${escapeHtml(value)}" ${STATE.explorer.sort === value ? "selected" : ""}>${escapeHtml(label)}</option>`
  )).join("");
  const presets = [
    { id: "python-debug", label: t("python_debug"), query: { tagQuery: "python, debug", sort: supportsScores ? "quality_asc" : "sample_id_asc" } },
  ];
  if (supportsScores) {
    presets.unshift(
      { id: "quality", label: t("lowest_quality"), query: { sort: "quality_asc" } },
      { id: "value", label: t("lowest_value"), query: { sort: "value_asc" } },
      { id: "confidence", label: t("low_confidence"), query: { sort: "confidence_asc" } },
    );
    presets.push(
      { id: "flags", label: t("has_flags"), query: { hasFlags: true, sort: "quality_asc" } },
      { id: "long-multiturn", label: t("long_multiturn"), query: { turnCountMin: "8", sort: "turn_count_desc" } },
      { id: "low-coverage", label: t("low_coverage"), query: { observedTurnRatioMax: "0.50", sort: "observed_turn_ratio_asc" } },
      { id: "low-rarity-confidence", label: t("low_rarity_conf"), query: { rarityConfidenceMax: "0.60", sort: "rarity_confidence_asc" } },
    );
  }
  const status = STATE.explorer.busy
    ? t("status_scanning", { done: fmtInt(STATE.explorer.scopeDone), total: fmtInt(STATE.explorer.scopeTotal), rows: fmtInt(STATE.explorer.scanned), matches: fmtInt(STATE.explorer.matched) })
    : (STATE.explorer.status || t("status_ready_scan", { count: fmtInt(totalSamples) }));

  return section(
    t("sample_explorer"),
    `<div id="explorer-panel">
    <div class="explorer-toolbar">
      <div>
        <div class="explorer-summary">${escapeHtml(t("explorer_summary", { limit: fmtInt(STATE.explorer.limit) }))}</div>
        <div class="preset-row">${presets.map((preset) => `<button class="preset-btn" type="button" data-explorer-preset="${escapeHtml(preset.id)}">${escapeHtml(preset.label)}</button>`).join("")}</div>
      </div>
      <div class="result-count">${escapeHtml(status)}</div>
    </div>
    <div class="explorer-form">
      <div class="field"><label>${escapeHtml(t("tags_comma"))}</label><input id="explorer-tag-query" type="text" value="${escapeHtml(query.tagQuery)}" placeholder="python, debug"></div>
      <div class="field"><label>${escapeHtml(t("language"))}</label><input id="explorer-language" type="text" value="${escapeHtml(query.language)}" placeholder="python"></div>
      <div class="field"><label>${escapeHtml(dimensionLabel("intent"))}</label><input id="explorer-intent" type="text" value="${escapeHtml(query.intent)}" placeholder="debug"></div>
      <div class="field"><label>${escapeHtml(t("source_file"))}</label><input id="explorer-source-path" type="text" value="${escapeHtml(query.sourcePath)}" placeholder="multi_turn/coderforge"></div>
      <div class="field"><label>${escapeHtml(t("text_contains"))}</label><input id="explorer-text-query" type="text" value="${escapeHtml(query.textQuery)}" placeholder="OOM"></div>
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("flag"))}</label><input id="explorer-flag-query" type="text" value="${escapeHtml(query.flagQuery)}" placeholder="low-correctness"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("conv_value"))} ≥</label><input id="explorer-conv-value-min" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.convValueMin)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("conv_sel_min"))}</label><input id="explorer-conv-selection-min" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.convSelectionMin)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("conv_sel_max"))}</label><input id="explorer-conv-selection-max" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.convSelectionMax)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("peak_cplx_min"))}</label><input id="explorer-peak-complexity-min" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.peakComplexityMin)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("turns_min"))}</label><input id="explorer-turn-count-min" type="number" min="1" step="1" value="${escapeHtml(query.turnCountMin)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("observed_ratio_min"))}</label><input id="explorer-observed-turn-ratio-min" type="number" min="0" max="1" step="0.05" value="${escapeHtml(query.observedTurnRatioMin)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("observed_ratio_max"))}</label><input id="explorer-observed-turn-ratio-max" type="number" min="0" max="1" step="0.05" value="${escapeHtml(query.observedTurnRatioMax)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("rarity_conf_min"))}</label><input id="explorer-rarity-confidence-min" type="number" min="0" max="1" step="0.05" value="${escapeHtml(query.rarityConfidenceMin)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("rarity_conf_max"))}</label><input id="explorer-rarity-confidence-max" type="number" min="0" max="1" step="0.05" value="${escapeHtml(query.rarityConfidenceMax)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("value_ge"))}</label><input id="explorer-min-value" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.minValue)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("value_le"))}</label><input id="explorer-max-value" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.maxValue)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("quality_le"))}</label><input id="explorer-max-quality" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.maxQuality)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("selection_ge"))}</label><input id="explorer-min-selection" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.minSelection)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("selection_le"))}</label><input id="explorer-max-selection" type="number" min="1" max="10" step="0.1" value="${escapeHtml(query.maxSelection)}"></div>` : ""}
      ${supportsScores ? `<div class="field"><label>${escapeHtml(t("confidence_le"))}</label><input id="explorer-max-confidence" type="number" min="0" max="1" step="0.01" value="${escapeHtml(query.maxConfidence)}"></div>` : ""}
      <div class="field"><label>${escapeHtml(t("thinking_mode"))}</label><select id="explorer-thinking-mode"><option value="">${escapeHtml(t("any"))}</option><option value="fast" ${query.thinkingMode === "fast" ? "selected" : ""}>${escapeHtml(thinkingModeLabel("fast"))}</option><option value="slow" ${query.thinkingMode === "slow" ? "selected" : ""}>${escapeHtml(thinkingModeLabel("slow"))}</option></select></div>
      <div class="field"><label>${escapeHtml(t("sort"))}</label><select id="explorer-sort">${sortOptions}</select></div>
      ${supportsScores ? `<div class="field-check"><input id="explorer-has-flags" type="checkbox" ${query.hasFlags ? "checked" : ""}><label for="explorer-has-flags">${escapeHtml(t("only_samples_with_flags"))}</label></div>` : ""}
      <div class="field-check"><input id="explorer-include-inherited" type="checkbox" ${query.includeInherited ? "checked" : ""}><label for="explorer-include-inherited">${escapeHtml(t("include_inherited_labels"))}</label></div>
    </div>
    <div class="explorer-actions">
      <button class="action-btn" type="button" id="explorer-run">${STATE.explorer.busy ? escapeHtml(t("scanning")) : escapeHtml(t("run_query"))}</button>
      <button class="action-btn" type="button" id="explorer-reset">${escapeHtml(t("reset"))}</button>
    </div>
    ${renderExplorerActiveFilters(query)}
    <div class="result-meta">
      <div class="result-count">${escapeHtml(t("current_scope_summary", { samples: fmtInt(totalSamples), files: fmtInt(candidateFileCount), limit: fmtInt(STATE.explorer.limit), sort: (availableSorts.find(([value]) => value === STATE.explorer.sort) || ["", STATE.explorer.sort])[1] }))}</div>
      <div class="result-count">${escapeHtml(t("matches_retained", { matches: fmtInt(STATE.explorer.matched), rows: fmtInt(STATE.explorer.scanned) }))}</div>
    </div>
    ${renderExplorerResults(STATE.explorer.results)}
    ${renderExplorerDrawer()}</div>`,
    `${fmtInt(totalSamples)} ${escapeHtml(t("slices"))}`,
    true,
  );
}

function renderScopeContent(scope) {
  const chunks = [];
  chunks.push(renderChildren(scope));
  chunks.push(renderPass1(scope.pass1));
  chunks.push(renderPass2(scope.pass2));
  chunks.push(renderExplorer(scope));
  chunks.push(renderConversations(scope.conversation));
  const html = chunks.filter(Boolean).join("");
  return html || `<div class="empty">${escapeHtml(t("no_dashboard_data"))}</div>`;
}

async function renderScope() {
  const scope = getCurrentScope();
  if (!scope) return;
  await ensureScopeDetail(scope.id);

  document.getElementById("scope-title").textContent = scope.label;
  document.getElementById("scope-path").textContent = scope.path || t("global_overview");
  renderBreadcrumbs(scope);
  document.getElementById("scope-content").innerHTML = renderScopeContent(scope);

  document.querySelectorAll("[data-jump]").forEach((node) => {
    node.addEventListener("click", () => setScope(node.getAttribute("data-jump")));
  });
  document.querySelectorAll("[data-file-sort]").forEach((node) => {
    node.addEventListener("click", () => toggleFileRankingSort(node.getAttribute("data-file-sort")));
  });
  document.querySelectorAll("[data-keep-rate-threshold]").forEach((node) => {
    node.addEventListener("click", () => {
      STATE.keepRateThreshold = normalizedKeepRateThreshold(node.getAttribute("data-keep-rate-threshold"));
      persistDashboardState();
      renderScope();
    });
  });
  document.querySelectorAll("[data-tag-bar-mode]").forEach((node) => {
    node.addEventListener("click", () => {
      STATE.tagBarMode = node.getAttribute("data-tag-bar-mode") || "relative";
      persistDashboardState();
      renderScope();
    });
  });
  document.querySelectorAll("[data-aggregation-mode]").forEach((node) => {
    node.addEventListener("click", () => {
      STATE.aggregationMode = node.getAttribute("data-aggregation-mode") || "sample";
      persistDashboardState();
      renderHero();
      renderTree();
      renderScope();
    });
  });
  document.querySelectorAll("[data-locale]").forEach((node) => {
    node.addEventListener("click", () => {
      setLocale(node.getAttribute("data-locale") || "en");
    });
  });
  const aggregationInfoBtn = document.getElementById("aggregation-info-btn");
  if (aggregationInfoBtn) {
    aggregationInfoBtn.addEventListener("click", () => {
      const wrap = document.getElementById("aggregation-info-wrap");
      if (wrap) wrap.classList.toggle("open");
    });
  }
  const explorerRun = document.getElementById("explorer-run");
  if (explorerRun) {
    explorerRun.addEventListener("click", async () => {
      STATE.explorer.sort = (document.getElementById("explorer-sort") || {}).value || STATE.explorer.sort;
      await runExplorerQuery();
    });
  }
  const explorerReset = document.getElementById("explorer-reset");
  if (explorerReset) {
    explorerReset.addEventListener("click", () => {
      resetExplorerView(true);
      persistDashboardState();
      renderScope();
    });
  }
  const explorerSort = document.getElementById("explorer-sort");
  if (explorerSort) {
    explorerSort.addEventListener("change", () => {
      STATE.explorer.sort = explorerSort.value;
      persistDashboardState();
      if (STATE.explorer.results.length) {
        STATE.explorer.results.sort((left, right) => compareExplorerRows(left, right, STATE.explorer.sort));
        renderScope();
      }
    });
  }
  document.querySelectorAll("[data-explorer-patch]").forEach((node) => {
    node.addEventListener("click", async () => {
      const raw = node.getAttribute("data-explorer-patch");
      if (!raw) return;
      try {
        const shouldScrollToExplorer = node.hasAttribute("data-extension-drilldown");
        applyExplorerQueryPatch(JSON.parse(raw));
        await runExplorerQuery();
        if (shouldScrollToExplorer) scrollExplorerIntoView();
      } catch (error) {
        console.warn("Failed to apply explorer patch", error);
      }
    });
  });
  document.querySelectorAll("[data-explorer-remove-tag]").forEach((node) => {
    node.addEventListener("click", async () => {
      const token = node.getAttribute("data-explorer-remove-tag");
      if (!token) return;
      STATE.explorer.queryModel = {
        ...STATE.explorer.queryModel,
        tagQuery: removeCsvValue(STATE.explorer.queryModel.tagQuery, token),
      };
      persistDashboardState();
      await runExplorerQuery();
      scrollExplorerIntoView();
    });
  });
  document.querySelectorAll("[data-explorer-preset]").forEach((node) => {
    node.addEventListener("click", async () => {
      const presetId = node.getAttribute("data-explorer-preset");
      const presetMap = {
        "python-debug": { tagQuery: "python, debug", sort: scopeHasPass2Data(getCurrentScope()) ? "quality_asc" : "sample_id_asc" },
        quality: { sort: "quality_asc" },
        value: { sort: "value_asc" },
        confidence: { sort: "confidence_asc" },
        flags: { hasFlags: true, sort: "quality_asc" },
        "long-multiturn": { turnCountMin: "8", sort: "turn_count_desc" },
        "low-coverage": { observedTurnRatioMax: "0.50", sort: "observed_turn_ratio_asc" },
        "low-rarity-confidence": { rarityConfidenceMax: "0.60", sort: "rarity_confidence_asc" },
      };
      if (presetMap[presetId]) setExplorerPreset(presetMap[presetId]);
      await runExplorerQuery();
    });
  });
  document.querySelectorAll("[data-explorer-detail]").forEach((node) => {
    node.addEventListener("click", async () => {
      await openExplorerDetail(
        node.getAttribute("data-explorer-detail"),
        node.getAttribute("data-explorer-scope"),
        node.getAttribute("data-explorer-detail-chunk"),
      );
    });
  });
  document.querySelectorAll("[data-explorer-close]").forEach((node) => {
    node.addEventListener("click", closeExplorerDetail);
  });

  const parentBtn = document.getElementById("go-parent");
  parentBtn.disabled = !scope.parent_id;
  parentBtn.onclick = () => {
    if (scope.parent_id) setScope(scope.parent_id);
  };
  const globalBtn = document.getElementById("go-global");
  globalBtn.disabled = scope.id === DATA.root_id;
  globalBtn.onclick = () => {
    if (scope.id !== DATA.root_id) setScope(DATA.root_id);
  };
}

async function render() {
  renderChromeText();
  renderHero();
  renderTree();
  await renderScope();
}

document.getElementById("scope-search").addEventListener("input", (event) => {
  STATE.query = event.target.value.trim().toLowerCase();
  renderTree();
});

document.querySelectorAll(".filter-btn").forEach((button) => {
  button.addEventListener("click", () => {
    STATE.filter = button.getAttribute("data-filter");
    document.querySelectorAll(".filter-btn").forEach((node) => node.classList.toggle("active", node === button));
    renderTree();
  });
});

STATE.sidebarCollapsed = loadSidebarPreference();
applySidebarState();
document.getElementById("sidebar-toggle").addEventListener("click", toggleSidebar);

window.addEventListener("hashchange", () => {
  restoreDashboardState();
  render();
});

restoreDashboardState();
await ensureScopeDetail(STATE.currentId);
await render();
if (explorerQueryHasFilters(STATE.explorer.queryModel) && scopeSupportsExplorer(getCurrentScope())) {
  await runExplorerQuery();
}
})();
