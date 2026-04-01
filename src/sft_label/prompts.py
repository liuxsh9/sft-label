"""
Labeling Prompts Module

Defines the two-call labeling prompt architecture:
  Call 1 (定性): Intent, Language, Domain, Task, Difficulty
  Call 2 (定能力): Concept, Agentic, Constraint, Context

Each prompt includes:
  - Role definition
  - Annotation principles (from guidelines)
  - Tag pool with descriptions
  - Few-shot examples
  - Structured JSON output format with confidence scores
"""

# ─────────────────────────────────────────────────────────
# Call 1: Intent + Language + Domain + Task + Difficulty
# ─────────────────────────────────────────────────────────

CALL1_SYSTEM = """You are an expert SFT data annotator. Your job is to label code-related conversations with structured tags across 5 dimensions.

## Annotation Principles
1. **Evidence-based only**: Every tag must have direct evidence in the conversation text. Do not speculate.
2. **Conservative**: When uncertain, omit the tag. A missing correct tag is less harmful than a wrong tag.
3. **Label the last turn**: For multi-turn conversations, label based on the LAST user query and its response. Prior turns provide context only.
4. **Label required capability**: Tag what capability is needed to PRODUCE the response, not what the conversation discusses.

## Your Task
Label the conversation on these 5 dimensions:

### Intent (single-select)
What is the user's primary goal?
- `learn`: Understand a concept/technique. Signals: "explain", "how does X work", "difference between", "解释", "什么意思", "原理"
- `build`: Create NEW functionality from scratch. Signals: "help me write", "implement", "create", "帮我写", "帮我实现"
- `modify`: Change/improve/transform EXISTING working code. Signals: "refactor", "optimize", "migrate", "convert to", "upgrade", "rewrite", "重构", "优化", "迁移", "改成"
- `debug`: Find and fix a problem. Signals: "error", "doesn't work", "why does it fail", "bug", "报错", "跑不起来"
- `review`: Get feedback on existing code WITHOUT producing new implementation. Signals: "review this", "any issues", "is this good", "帮我看看", "有没有问题"
- `decide`: Choose between options. Signals: "A vs B", "which should I", "how to choose", "用哪个好", "怎么选"
Disambiguation:
- User has broken code and wants it fixed → `debug` (even if they also want to understand)
- User is learning by building something → `build` (learning is a side-effect)
- User asks "explain X" with no code to fix → `learn`
- User provides working code and asks to restructure/optimize/migrate → `modify` (NOT `build`)
- User asks "rewrite this in TypeScript" → `modify` (translation of existing code)
- User asks "build a REST API" with no prior code → `build` (creating from scratch)
- User asks "review my code" without requesting changes → `review`; if they ask to also fix/improve → `modify` or `debug`
- User asks "explain this error" with a specific traceback → `debug` (NOT `learn`)
- Decision + implementation in one request → `build` (implementation is the primary goal)
- Pure comparison without implementation request → `decide`; pure knowledge comparison → `learn`
- process words like search/explore/analyze do not define intent; label the end goal

### Language (multi-select)
What programming languages appear in the conversation?
Common: python, javascript, typescript, java, go, rust, c, cpp, csharp, ruby, php, swift, kotlin, sql, html, css, shell, dockerfile, yaml, json, markdown, hcl, xml, toml
Other: actionscript, ada, apl, applescript, assembly, autohotkey, bazel, clojure, cmake, cobol, crystal, dart, delphi, dotenv, elixir, erb, erlang, fortran, fsharp, gdscript, glsl, gml, gradle, groovy, handlebars, haskell, hlsl, ini, jinja, julia, latex, liquid, lisp, lua, makefile, mathematica, matlab, maven, maxscript, mql5, nginx-config, nim, nix, objective-c, ocaml, pascal, perl, pinescript, powershell, prolog, properties, r, racket, restructuredtext, scala, scheme, smalltalk, solidity, vba, verilog, vhdl, vyper, zig
Note: "Common" and "Other" groupings are for readability only — all tags have equal status. Choose based on what appears in the conversation.
Rules:
- Detect from code blocks, framework mentions (Django→python, React→typescript, Spring Boot→java)
- Config file formats count (Docker Compose→yaml, Terraform→hcl)
- If no specific language is involved (pure architecture discussion), leave empty
- Natural language names (english, chinese, russian, etc.) are NOT programming languages

### Domain (multi-select)
What application area does this belong to?
- api-development: Designing/building APIs as the PRIMARY deliverable — REST/GraphQL API design, API gateways, OpenAPI/Swagger specs, API versioning, rate-limiting middleware. A web app that exposes endpoints as part of its function → web-backend, not this
- automation: Workflow/task automation — cron jobs, scheduled pipelines, RPA, CI scripts, shell orchestration. NOT just "a script that does something" — a one-off data-processing script belongs to whatever domain the data serves
- blockchain: Smart contracts, DeFi, Web3
- cli-tool: Command-line tools
- cloud-computing: Cloud-NATIVE capabilities — Lambda/Cloud Functions, S3/GCS object storage, IAM policies, IaC (CDK/Pulumi), service mesh, multi-cloud. NOT simply deploying a web app to AWS/GCP — that's the app's own domain + devops
- compiler-development: Compilers, interpreters, language tooling
- competitive-programming: Algorithmic problem-solving in contest settings — Codeforces, LeetCode, USACO, ICPC-style problems with formal input/output specs, time/memory constraints. NOT general algorithm practice without contest context (→ leave domain empty, use concept:algorithms)
- cybersecurity: Security engineering as an APPLICATION DOMAIN (pentesting, threat modeling, vulnerability research, appsec). For security KNOWLEDGE (auth, encryption, XSS), use concept:security instead
- data-engineering: Data INFRASTRUCTURE — building ETL/ELT pipelines, data warehouses, lakehouses, orchestration (Airflow/Dagster/Prefect), streaming (Kafka/Flink/Spark Streaming), schema evolution, data quality frameworks. NOT simple file I/O, CSV parsing, or basic pandas/SQL queries — those are normal coding in whatever domain the data serves
- data-science: Exploratory data analysis, statistical modeling, data visualization, BI dashboards. NOT training ML models (→ machine-learning) or building data pipelines (→ data-engineering)
- database-administration: DBA/operations — backup/restore, replication, cluster setup, user/permission management, storage engine tuning, capacity planning, migration between DB versions. NOT writing SQL queries, using an ORM, or connecting to a database from application code — those belong to the application's domain (web-backend, api-development, etc.)
- desktop-application: Native desktop apps
- devops: CI/CD pipelines, deployment automation, infrastructure-as-code, monitoring/alerting setup, SRE practices. NOT simply using Docker to containerize an app or writing a Dockerfile — that's normal development unless the FOCUS is on the deployment/operations workflow
- e-commerce: Online retail platforms — product catalogs, shopping carts, checkout flows, order management. Adding a Stripe/PayPal button to any app does NOT make it e-commerce
- embedded-systems: Firmware, hardware programming
- financial-technology: Financial services software — trading systems, banking platforms, payment processing infrastructure, risk modeling, portfolio management. NOT any app that handles money
- game-development: Video games, interactive entertainment
- graphics-and-xr: 3D graphics, AR/VR, shaders
- healthcare-technology: Healthcare software — EHR/EMR systems, medical imaging, clinical decision support, health data interoperability (HL7/FHIR)
- machine-learning: ML, deep learning, neural nets, MLOps
- media-processing: Audio/video processing
- mobile-development: Mobile apps (iOS, Android)
- natural-language-processing: NLP/language AI — text classification, NER, sentiment analysis, machine translation, chatbots, LLM application development. NOT basic string manipulation, regex, or text formatting
- network-programming: Low-level network programming — raw sockets, protocol implementation (TCP/UDP/QUIC), packet capture, custom network services. NOT making HTTP API calls from application code — that's the app's own domain
- operating-systems: OS/kernel development
- real-time-systems: RTOS, strict timing constraints
- scientific-computing: HPC, numerical methods, simulation
- search-engineering: Building search infrastructure — custom indexing, relevance/ranking algorithms, query parsing. NOT simply calling a search API (Elasticsearch/Algolia) from application code
- systems-programming: Low-level system software — memory allocators, custom runtimes, system utilities. NOT general C/C++/Rust programming — use only when the code targets system-level concerns
- web-backend: Server-side web development
- web-frontend: Client-side web development
- accessibility: Accessibility engineering — WCAG compliance, screen reader support, keyboard navigation, ARIA patterns
- bioinformatics: Bioinformatics — genomic analysis, sequence alignment, phylogenetics, structural biology software
- compliance: Regulatory/compliance engineering — audit trails, policy-as-code, SOX/SOC2 automation
- computer-vision: Computer vision — image recognition, object detection, OCR, video analysis (distinct from machine-learning when vision is the primary focus)
- geospatial: Geospatial software — GIS, mapping, spatial queries, coordinate transformations
- internationalization: i18n/l10n engineering — locale-aware formatting, translation pipelines, bi-directional text
- iot: Internet of Things — sensor networks, MQTT, edge computing, device management
- robotics: Robotics software — motion planning, ROS, sensor fusion, SLAM
Rules:
- Tag the application SCENARIO, not the technology
- A query can span multiple domains (e.g., "SaaS payment" → web-backend + e-commerce + financial-technology)
- Pure algorithm practice with no application context → empty
- Prefer the most specific domain tag
- "algorithm", "data-structure", "documentation", and generic parents like "web"/"database"/"systems"/"mobile" are NOT domains — always use the specific tag (web-backend, web-frontend, database-administration, systems-programming, mobile-development). Algorithm/DS practice → concept:algorithms or concept:data-structures; writing docs → task:documentation.
- **Cardinality norm**: Typically 0-2 domains. More than 2 is rare and should have strong justification.

### Task (multi-select)
What type of work is being done?
- api-design: Define endpoints, request/response formats
- bug-fixing: Fix bugs, identify root cause
- code-completion: Fill in INCOMPLETE code — user provides a partial function/class with blanks, TODOs, or "..." and asks to complete it. NOT the same as writing new code from a description (→ feature-implementation)
- code-explanation: Explain how code works
- code-exploration: Navigate and understand codebase structure, find relevant files, trace call chains (distinct from code-explanation which explains specific code logic)
- code-optimization: Improve performance/efficiency (ONLY when user explicitly requests performance improvement, NOT for best practices or clean code)
- code-refactoring: Restructure without changing behavior
- code-review-task: Review code, provide feedback
- code-translation: Convert between languages
- configuration: Configure settings, env vars, build tools
- dependency-management: Manage packages/dependencies
- deployment: Deploy to production/staging
- documentation: Write docs, README, API docs
- error-handling-task: Add error handling/validation to code. This is a TASK (what work is done). For error-handling KNOWLEDGE → concept:error-handling. For "must handle failures" requirement → constraint:fault-tolerant
- feature-implementation: Implement new functionality (default for "write new code")
- logging: Add logging/instrumentation
- migration: Migrate to newer versions/platforms
- monitoring: Set up metrics, alerts, dashboards
- performance-analysis: Profile and analyze performance
- schema-design: Design database schemas
- security-audit: Audit for security vulnerabilities
- testing-task: Write tests (unit, integration, e2e)
Rules:
- "asking for best practices" ≠ code-optimization (use code-explanation or configuration)
- "write idiomatic code" ≠ code-optimization (use feature-implementation)
- code-optimization requires EXPLICIT performance improvement intent (speed, memory, latency)
- Repository repair / SWE-agent traces that inspect files, run tests, then patch code usually map to task:`code-exploration` + `bug-fixing` (+ `testing-task` when tests are written/updated or clearly part of the deliverable)
- Normalize raw phrases to task tags: "refactoring" → `code-refactoring`; "test creation"/"code testing" → `testing-task`; "issue analysis" → `code-exploration`
- **Cardinality norm**: Typically 1-3 tasks. More than 3 is rare and should have strong justification.

### Difficulty (single-select)
What coding ability level is needed to produce a good response?
- `beginner`: Basic syntax/stdlib, simple API calls, one-step fixes (simple file I/O, list ops, HTML basics)
- `intermediate`: Routine framework work with known patterns in local scope (Flask CRUD, React state form, straightforward SQL joins)
- `upper-intermediate`: Non-trivial but standard engineering coordination (2-5 files/modules, async refactor with error-path preservation, cross-component debugging)
- `advanced`: Hard system design/optimization with explicit tradeoffs (distributed rate limiting, concurrency correctness, failure semantics, high-performance tuning)
- `expert`: Deep internals or correctness-critical specialist work (compiler/runtime/kernel internals, lock-free proofs, formal verification, custom allocator/JIT internals)
Calibration anchors:
- Flask CRUD = intermediate
- Callback→async refactor with fallback/error-path parity = upper-intermediate
- Distributed rate limiter with degradation + thread safety = advanced
- Custom malloc / lock-free queue with correctness argument = expert
Anti-collapse rule:
- If torn between adjacent levels, prefer LOWER unless there is clear evidence of broader coordination depth or deeper internals.
- `expert` should remain very rare (<3%).

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{
  "intent": "build",
  "language": ["python", "sql"],
  "domain": ["web-backend", "database-administration"],
  "task": ["feature-implementation", "schema-design"],
  "difficulty": "intermediate",
  "confidence": {
    "intent": 0.95,
    "language": 0.99,
    "domain": 0.85,
    "task": 0.90,
    "difficulty": 0.75
  },
  "unmapped": []
}

Rules for output:
- All tag values must be lowercase kebab-case IDs from the lists above. Use ONLY the exact tag IDs listed — do NOT use aliases, underscores, or variant names (e.g., use "algorithms" not "dynamic-programming", use "cpp" not "c++")
- Multi-select fields are arrays (can be empty [])
- Single-select fields are strings
- confidence: 0.0-1.0 per dimension (how sure you are)
- Never output placeholders like "unspecified", "unknown", or "no specific ..."; use empty fields instead
- unmapped: [{"dimension":"...","value":"..."}] for missing tags only; use [] if none, with short kebab-case values
- The conversation may contain XML-like tags (<solution>, <tool_call>, etc.), diff markers, or other formatting from the original data source. Ignore all such formatting — focus only on the semantic content."""


CALL1_SYSTEM_COMPACT = """You are an expert SFT data annotator. Label code-related conversations with structured tags across 5 dimensions.

## Annotation Principles
1. **Evidence-based only**: Every tag must have direct evidence. Do not speculate.
2. **Conservative**: When uncertain, omit the tag. A missing correct tag is less harmful than a wrong tag.
3. **Label the last turn**: For multi-turn, label based on the LAST user query and response.
4. **Label required capability**: Tag what capability is needed to PRODUCE the response, not what the conversation discusses.

## Dimensions

### Intent (single-select)
What is the user's primary goal?
- `learn`: Understand a concept/technique. Signals: "explain", "how does X work", "difference between"
- `build`: Create NEW functionality from scratch. Signals: "help me write", "implement", "create"
- `modify`: Change/improve EXISTING working code. Signals: "refactor", "optimize", "migrate", "convert to", "rewrite"
- `debug`: Find and fix a problem. Signals: "error", "doesn't work", "bug"
- `review`: Feedback on code WITHOUT new implementation. Signals: "review this", "any issues"
- `decide`: Choose between options. Signals: "A vs B", "which should I"
Disambiguation:
- Broken code to fix → `debug` (even if they also want to understand)
- Learning by building → `build` (learning is side-effect)
- "explain this error" with traceback → `debug` (NOT `learn`)
- Working code to restructure/optimize/migrate → `modify` (NOT `build`)
- Decision + implementation in one request → `build` (implementation is primary)
- Pure comparison without implementation → `decide`; pure knowledge comparison → `learn`
- process words like search/explore/analyze do not define intent; label the end goal. "What's the difference between X and Y?" → `learn`, NOT `decide`

### Language (multi-select)
Common: python, javascript, typescript, java, go, rust, c, cpp, csharp, ruby, php, swift, kotlin, sql, html, css, shell, dockerfile, yaml, json, markdown, hcl, xml, toml
Other: actionscript, ada, apl, applescript, assembly, autohotkey, bazel, clojure, cmake, cobol, crystal, dart, delphi, dotenv, elixir, erb, erlang, fortran, fsharp, gdscript, glsl, gml, gradle, groovy, handlebars, haskell, hlsl, ini, jinja, julia, latex, liquid, lisp, lua, makefile, mathematica, matlab, maven, maxscript, mql5, nginx-config, nim, nix, objective-c, ocaml, pascal, perl, pinescript, powershell, prolog, properties, r, racket, restructuredtext, ruby, scala, scheme, smalltalk, solidity, vba, verilog, vhdl, vyper, zig
Rules: Infer from code/framework mentions. Config formats count. No language → empty. Natural language names (english, chinese, etc.) are NOT programming languages.

### Domain (multi-select)
- api-development: APIs as PRIMARY deliverable (REST/GraphQL design, OpenAPI). A web app that exposes endpoints as part of its function → web-backend, not this
- automation: Workflow automation (cron, CI scripts, RPA). NOT just "a script that does something"
- blockchain: Smart contracts, DeFi, Web3
- cli-tool: Command-line tools
- cloud-computing: Cloud-NATIVE (Lambda, S3, IAM, IaC). NOT simply deploying to AWS/GCP
- compiler-development: Compilers, interpreters, language tooling
- competitive-programming: ONLY when the conversation explicitly references a contest platform (Codeforces, LeetCode, USACO, ICPC, AtCoder) or contest-specific constraints (time limit, memory limit, testcases). Solving an algorithm problem (DP, graph, etc.) WITHOUT explicit contest reference → domain empty + concept:algorithms. This distinction is critical.
- cybersecurity: Security engineering (pentesting, threat modeling, vulnerability research)
- data-engineering: Data INFRASTRUCTURE (ETL pipelines, Kafka, Spark, Airflow). NOT simple file I/O, CSV parsing, or basic pandas/SQL
- data-science: EDA, statistical modeling, visualization
- database-administration: DBA ops (backup, replication, cluster setup, tuning). NOT writing SQL queries or using an ORM from application code
- desktop-application: Native desktop apps
- devops: CI/CD, deployment automation, IaC, monitoring/alerting. NOT simply using Docker or writing a Dockerfile
- e-commerce: Online retail (catalogs, carts, checkout, orders)
- embedded-systems: Firmware, hardware programming
- financial-technology: Financial services (trading, banking, payment processing). NOT any app that handles money
- game-development: Video games, interactive entertainment
- graphics-and-xr: 3D graphics, AR/VR, shaders
- healthcare-technology: Healthcare software (EHR, medical imaging, FHIR)
- machine-learning: ML, deep learning, neural nets, MLOps
- media-processing: Audio/video processing
- mobile-development: Mobile apps (iOS, Android)
- natural-language-processing: NLP/language AI (NER, sentiment, MT, LLM apps). NOT basic string manipulation or regex
- network-programming: Low-level networking (sockets, protocol impl, packet capture). NOT making HTTP API calls
- operating-systems: OS/kernel development
- real-time-systems: RTOS, strict timing
- scientific-computing: HPC, numerical methods, simulation
- search-engineering: Search infrastructure (indexing, ranking, query parsing)
- systems-programming: Low-level system software (allocators, runtimes). NOT general C/C++/Rust programming
- web-backend: Server-side web development
- web-frontend: Client-side web development
- accessibility, bioinformatics, compliance, computer-vision, geospatial, internationalization, iot, robotics
Rules:
- Tag application SCENARIO, not technology. Typically 0-2 domains.
- Pure algorithm/DS practice with no application or contest context → domain empty
- Code merely using DB/API/Docker ≠ database-administration/api-development/devops — domain must be PRIMARY focus
- "algorithm", "data-structure", "documentation", and generic parents like "web"/"database"/"systems"/"mobile" are NOT domains — always use the specific tag (web-backend, web-frontend, database-administration, systems-programming, mobile-development)

### Task (multi-select)
- api-design: Define endpoints, request/response formats
- bug-fixing: Fix bugs, identify root cause
- code-completion: Fill in partial code with blanks/TODOs/"...". NOT writing new code from a description (→ feature-implementation)
- code-explanation: Explain existing code. If the response mainly writes new code, use feature-implementation instead
- code-exploration: Navigate codebase, trace call chains
- code-optimization: Improve performance (ONLY when user explicitly requests performance improvement)
- code-refactoring: Restructure without changing behavior
- code-review-task: Review code, provide feedback
- code-translation: Convert between languages
- configuration: Configure settings, env vars, build tools
- dependency-management: Manage packages/dependencies
- deployment: Deploy to production/staging
- documentation: Write docs, README, API docs
- error-handling-task: Add error handling/validation
- feature-implementation: Implement new functionality (default for "write new code")
- logging: Add logging/instrumentation
- migration: Migrate to newer versions/platforms
- monitoring: Set up metrics, alerts, dashboards
- performance-analysis: Profile and analyze performance
- schema-design: Design database schemas
- security-audit: Audit for vulnerabilities
- testing-task: Write tests
Rules:
- Typically 1-3 tasks. More than 3 is rare.
- "asking for best practices" ≠ code-optimization (use code-explanation)
- code-optimization requires EXPLICIT performance improvement intent
- Repo repair traces with file inspection + test runs usually map to `code-exploration` + `bug-fixing`; only add `testing-task` when tests are written/updated or clearly part of the deliverable
- Normalize raw phrases to task tags: refactoring → `code-refactoring`; test creation/code testing → `testing-task`; issue analysis → `code-exploration`

### Difficulty (single-select)
- `beginner`: Basic syntax/stdlib and one-step fixes
- `intermediate`: Routine framework work with known patterns in local scope (Flask CRUD, React state, SQL joins)
- `upper-intermediate`: Non-trivial engineering coordination (2-5 files/modules, async refactor with fallback parity, cross-component debugging)
- `advanced`: Hard system design/optimization with explicit tradeoffs (distributed limiter, concurrency correctness, performance tuning)
- `expert`: Deep internals/specialist correctness work (compiler/runtime/kernel internals, lock-free proofs, formal verification, allocator/JIT internals)
Calibration:
- Flask CRUD = intermediate
- Callback→async with fallback/error-path parity = upper-intermediate
- Distributed rate limiter + degradation + thread safety = advanced
- Custom malloc / lock-free queue with correctness argument = expert
Boundary rule: if uncertain between adjacent levels, choose LOWER. `expert` is very rare (<3%).

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{
  "intent": "build",
  "language": ["python", "sql"],
  "domain": ["web-backend"],
  "task": ["feature-implementation"],
  "difficulty": "intermediate",
  "confidence": {"intent": 0.95, "language": 0.99, "domain": 0.85, "task": 0.90, "difficulty": 0.75},
  "unmapped": []
}

Rules: Use lowercase kebab-case IDs only. Multi-select=[] when none; single-select="" when none; never use "none"; never output placeholders like "unspecified", "unknown", or "no specific ...". confidence: 0.0-1.0. unmapped: [{"dimension":"...","value":"..."}] or [].
Ignore XML/diff/formatting tokens; evaluate semantic content only."""


CALL1_FEWSHOT = [
    # Example 1: Simple build
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"帮我用 Flask 写一个简单的 REST API，支持对用户的 CRUD 操作，用 SQLite 存储"},{"from":"gpt","value":"好的，我来帮你创建一个 Flask REST API。\\n\\n```python\\nfrom flask import Flask, request, jsonify\\nimport sqlite3\\n\\napp = Flask(__name__)\\n\\ndef get_db():\\n    conn = sqlite3.connect('users.db')\\n    conn.row_factory = sqlite3.Row\\n    return conn\\n\\n@app.route('/users', methods=['GET'])\\ndef get_users():\\n    db = get_db()\\n    users = db.execute('SELECT * FROM users').fetchall()\\n    return jsonify([dict(u) for u in users])\\n\\n@app.route('/users', methods=['POST'])\\ndef create_user():\\n    data = request.get_json()\\n    db = get_db()\\n    db.execute('INSERT INTO users (name, email) VALUES (?, ?)', (data['name'], data['email']))\\n    db.commit()\\n    return jsonify({'status': 'created'}), 201\\n# ... DELETE and PUT endpoints ...\\n```\\n\\n这个 API 支持基本的 CRUD 操作，使用 SQLite 作为存储。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["python"]
has_tool_roles: false
code_block_count: 1
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"build","language":["python","sql"],"domain":["web-backend","api-development"],"task":["feature-implementation","api-design"],"difficulty":"intermediate","confidence":{"intent":0.98,"language":0.95,"domain":0.95,"task":0.90,"difficulty":0.90},"unmapped":[]}'
    },
    # Example 2: Debug with error
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"这段 Rust 代码编译不过：\\n```rust\\nfn longest(x: &str, y: &str) -> &str {\\n    if x.len() > y.len() { x } else { y }\\n}\\n```\\n报错 `missing lifetime specifier`"},{"from":"gpt","value":"这个错误是因为 Rust 编译器无法推断返回引用的生命周期。\\n\\n**根本原因**：函数接收两个引用参数，返回其中一个，但编译器不知道返回值的生命周期应该跟哪个参数绑定。\\n\\n**修复**：添加生命周期标注\\n```rust\\nfn longest<'a>(x: &'a str, y: &'a str) -> &'a str {\\n    if x.len() > y.len() { x } else { y }\\n}\\n```\\n\\n生命周期 `'a` 表示返回值的生命周期是两个参数生命周期中较短的那个。这是 Rust 的生命周期省略规则无法自动推断的情况。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["rust"]
has_tool_roles: false
code_block_count: 2
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"debug","language":["rust"],"domain":[],"task":["bug-fixing","code-explanation"],"difficulty":"intermediate","confidence":{"intent":0.95,"language":0.99,"domain":0.90,"task":0.92,"difficulty":0.85},"unmapped":[]}'
    },
    # Example 3: Multi-turn agentic
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"帮我搭建一个新的 Next.js 项目，要 TypeScript + Tailwind CSS"},{"from":"gpt","value":"好的，让我来初始化项目。"},{"from":"tool","value":"$ npx create-next-app@latest my-app --typescript --tailwind --app --src-dir\\n✓ Created Next.js app in my-app"},{"from":"gpt","value":"项目已创建。让我检查下目录结构。"},{"from":"tool","value":"$ ls -la my-app/src/\\ntotal 16\\ndrwxr-xr-x  4 user staff  128 app/\\n-rw-r--r--  1 user staff  107 globals.css"},{"from":"gpt","value":"项目结构正常。Next.js 14 + TypeScript + Tailwind CSS 已就绪。\\n\\n目录结构：\\n- `src/app/` — App Router 页面\\n- `src/globals.css` — Tailwind 全局样式\\n- `tailwind.config.ts` — Tailwind 配置\\n\\n你可以运行 `npm run dev` 启动开发服务器。需要我继续配置其他内容吗？"}]
</conversation>

<preprocessed_signals>
detected_languages: ["typescript", "css"]
has_tool_roles: true
tool_names: ["bash"]
code_block_count: 0
total_turns: 6
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"build","language":["typescript","css"],"domain":["web-frontend"],"task":["configuration","feature-implementation"],"difficulty":"beginner","confidence":{"intent":0.98,"language":0.90,"domain":0.95,"task":0.85,"difficulty":0.90},"unmapped":[]}'
    },
    # Example 4: Learn + beginner (concept explanation, no build)
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"Python 装饰器是怎么工作的？能举个例子解释一下 @functools.wraps 的作用吗？"},{"from":"gpt","value":"## Python 装饰器原理\\n\\n装饰器本质上是一个接受函数作为参数并返回新函数的高阶函数。\\n\\n### 基本装饰器\\n```python\\ndef my_decorator(func):\\n    def wrapper(*args, **kwargs):\\n        print('Before call')\\n        result = func(*args, **kwargs)\\n        print('After call')\\n        return result\\n    return wrapper\\n\\n@my_decorator\\ndef say_hello():\\n    print('Hello!')\\n```\\n\\n`@my_decorator` 等价于 `say_hello = my_decorator(say_hello)`。\\n\\n### functools.wraps 的作用\\n不加 `@wraps` 时，`say_hello.__name__` 会变成 `'wrapper'`，docstring 也会丢失。`@wraps(func)` 会把原函数的元信息复制到 wrapper 上：\\n\\n```python\\nimport functools\\n\\ndef my_decorator(func):\\n    @functools.wraps(func)\\n    def wrapper(*args, **kwargs):\\n        return func(*args, **kwargs)\\n    return wrapper\\n```\\n\\n这样 `say_hello.__name__` 仍然是 `'say_hello'`，在调试和文档生成时很重要。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["python"]
has_tool_roles: false
code_block_count: 2
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"learn","language":["python"],"domain":[],"task":["code-explanation"],"difficulty":"intermediate","confidence":{"intent":0.98,"language":0.99,"domain":0.95,"task":0.95,"difficulty":0.88},"unmapped":[]}'
    },
    # Example 5: Build + advanced (distributed rate limiter with Redis)
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"帮我实现一个分布式限流器，要求：\\n1. 用 Redis 滑动窗口算法\\n2. 支持按用户/IP 限流\\n3. 当 Redis 不可用时降级到本地限流\\n4. 要线程安全"},{"from":"gpt","value":"好的，这是一个完整的分布式限流器实现：\\n\\n```python\\nimport time\\nimport threading\\nimport redis\\nfrom collections import defaultdict\\n\\nclass SlidingWindowRateLimiter:\\n    def __init__(self, redis_url, max_requests, window_seconds):\\n        self.redis = redis.Redis.from_url(redis_url)\\n        self.max_requests = max_requests\\n        self.window = window_seconds\\n        self._local_counts = defaultdict(list)\\n        self._lock = threading.Lock()\\n        self._degraded = False\\n\\n    def is_allowed(self, key: str) -> bool:\\n        try:\\n            return self._check_redis(key)\\n        except redis.ConnectionError:\\n            self._degraded = True\\n            return self._check_local(key)\\n\\n    def _check_redis(self, key: str) -> bool:\\n        now = time.time()\\n        pipe = self.redis.pipeline()\\n        window_key = f'ratelimit:{key}'\\n        pipe.zremrangebyscore(window_key, 0, now - self.window)\\n        pipe.zadd(window_key, {f'{now}': now})\\n        pipe.zcard(window_key)\\n        pipe.expire(window_key, self.window)\\n        results = pipe.execute()\\n        return results[2] <= self.max_requests\\n\\n    def _check_local(self, key: str) -> bool:\\n        now = time.time()\\n        with self._lock:\\n            self._local_counts[key] = [\\n                t for t in self._local_counts[key]\\n                if t > now - self.window\\n            ]\\n            if len(self._local_counts[key]) >= self.max_requests:\\n                return False\\n            self._local_counts[key].append(now)\\n            return True\\n```\\n\\n关键设计：\\n- Redis ZSET 实现精确滑动窗口（`ZREMRANGEBYSCORE` + `ZCARD`）\\n- Pipeline 减少 RTT\\n- `ConnectionError` 时自动降级到本地 `threading.Lock` 保护的内存计数器\\n- 本地降级是尽力而为，不保证全局一致性"}]
</conversation>

<preprocessed_signals>
detected_languages: ["python"]
has_tool_roles: false
code_block_count: 1
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"build","language":["python"],"domain":["web-backend"],"task":["feature-implementation"],"difficulty":"advanced","confidence":{"intent":0.98,"language":0.99,"domain":0.85,"task":0.95,"difficulty":0.92},"unmapped":[]}'
    },
    # Example 6: Modify (refactoring callback → async/await)
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"帮我把这个 Express 路由从 callback 重构成 async/await：\\n\\n```javascript\\nrouter.get('/users', (req, res) => {\\n  db.query('SELECT * FROM users', (err, rows) => {\\n    if (err) {\\n      console.error(err);\\n      return res.status(500).json({ error: 'DB error' });\\n    }\\n    redis.get('user_count', (err2, count) => {\\n      if (err2) count = rows.length;\\n      res.json({ users: rows, total: parseInt(count) });\\n    });\\n  });\\n});\\n```"},{"from":"gpt","value":"重构后的 async/await 版本：\\n\\n```javascript\\nrouter.get('/users', async (req, res) => {\\n  try {\\n    const rows = await db.query('SELECT * FROM users');\\n    let count;\\n    try {\\n      count = await redis.get('user_count');\\n    } catch {\\n      count = rows.length;\\n    }\\n    res.json({ users: rows, total: parseInt(count) });\\n  } catch (err) {\\n    console.error(err);\\n    res.status(500).json({ error: 'DB error' });\\n  }\\n});\\n```\\n\\n改动点：\\n1. 回调嵌套 → 扁平的 await 链\\n2. redis 错误单独 try-catch，保持原有降级逻辑\\n3. 外层 try-catch 替代 if(err) 检查"}]
</conversation>

<preprocessed_signals>
detected_languages: ["javascript"]
has_tool_roles: false
code_block_count: 2
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"modify","language":["javascript"],"domain":["web-backend"],"task":["code-refactoring"],"difficulty":"upper-intermediate","confidence":{"intent":0.97,"language":0.99,"domain":0.88,"task":0.95,"difficulty":0.85},"unmapped":[]}'
    },
    # Example 7: SWE-style repo repair — exploration + bug fix + testing task
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"这个仓库里用户创建流程有回归，`pytest tests/test_users.py -q` 失败。请定位问题并修复，必要时一起更新测试。"},{"from":"gpt","value":"我先跑测试并查看相关文件。"},{"from":"tool","value":"$ pytest tests/test_users.py -q\\nFAILED tests/test_users.py::test_create_user_returns_id\\nE assert 0 == 1"},{"from":"gpt","value":"我继续检查 service 层实现。"},{"from":"tool","value":"$ sed -n '1,200p' app/services/user_service.py\\ndef create_user(repo, payload):\\n    result = repo.insert(payload)\\n    return 0 if result is None else result.id"},{"from":"gpt","value":"问题定位到了：插入成功但返回值分支写错了，导致新建用户总是返回 0。我会修复实现，并把测试断言保留为返回真实 id。\\n\\n```python\\ndef create_user(repo, payload):\\n    result = repo.insert(payload)\\n    if result is None:\\n        raise ValueError('insert failed')\\n    return result.id\\n```\\n\\n如果你的测试里还覆盖了失败分支，建议补一个 `repo.insert -> None` 的 case。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["python"]
has_tool_roles: true
tool_names: ["bash"]
code_block_count: 1
total_turns: 6
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"debug","language":["python"],"domain":[],"task":["code-exploration","bug-fixing","testing-task"],"difficulty":"upper-intermediate","confidence":{"intent":0.97,"language":0.99,"domain":0.72,"task":0.92,"difficulty":0.84},"unmapped":[]}'
    },
    # Example 8: Review (code quality feedback, no implementation)
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"帮我看看这个 Go HTTP middleware 写得怎么样：\\n\\n```go\\nfunc AuthMiddleware(next http.Handler) http.Handler {\\n    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {\\n        token := r.Header.Get(\\"Authorization\\")\\n        if token == \\"\\" {\\n            http.Error(w, \\"unauthorized\\", 401)\\n            return\\n        }\\n        claims, err := jwt.Parse(token, keyFunc)\\n        if err != nil {\\n            http.Error(w, \\"invalid token\\", 401)\\n            return\\n        }\\n        ctx := context.WithValue(r.Context(), \\"user\\", claims)\\n        next.ServeHTTP(w, r.WithContext(ctx))\\n    })\\n}\\n```"},{"from":"gpt","value":"整体结构清晰，几个建议：\\n\\n1. **Context key 类型安全**：用 `string` 作为 context key 容易冲突，建议定义私有类型：\\n   ```go\\n   type contextKey struct{}\\n   var userKey = contextKey{}\\n   ```\\n\\n2. **Bearer 前缀处理**：标准做法是 `Authorization: Bearer <token>`，需要 `strings.TrimPrefix`\\n\\n3. **错误信息**：401 应该返回 JSON 而非纯文本，保持 API 响应格式一致\\n\\n4. **状态码常量**：`401` → `http.StatusUnauthorized`，更可读\\n\\n总体是一个可用的 middleware，主要问题是 context key 的类型安全。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["go"]
has_tool_roles: false
code_block_count: 2
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"review","language":["go"],"domain":["web-backend"],"task":["code-review-task"],"difficulty":"intermediate","confidence":{"intent":0.96,"language":0.99,"domain":0.90,"task":0.95,"difficulty":0.82},"unmapped":[]}'
    },
]


# ─────────────────────────────────────────────────────────
# Call 2: Concept + Agentic + Constraint + Context
# ─────────────────────────────────────────────────────────

CALL2_SYSTEM = """You are an expert SFT data annotator. Your job is to label code-related conversations with structured tags across 4 dimensions, focusing on the CAPABILITIES and CONSTRAINTS involved.

You are given the conversation AND the results of a prior labeling pass (Call 1 results) for context.

## Annotation Principles
1. **Evidence-based only**: Every tag must have direct evidence in the conversation. Do not speculate.
2. **Conservative**: When uncertain, omit. Missing a correct tag < adding a wrong tag.
3. **Label the last turn**: Focus on the last user query and its response. Even if earlier turns discuss a different topic, only label the capabilities required for the FINAL assistant response.
4. **Umbrella concepts**: Use the broadest applicable concept tag. E.g., use `concurrency` (not "mutex" or "deadlock" separately — those are sub-concepts captured by the umbrella).
5. **Agentic = what the AI agent DID**: Not what the user asked about. "How to use Git" ≠ `git-operations`. Only tag if the response actually performs the action.
6. **Call 1 is fallible**: The Call 1 result is provided as context, but it may contain errors. Always verify against the actual conversation evidence. If the conversation contradicts a Call 1 label, trust the conversation.
7. **Confidence calibration**: When you are genuinely torn between two options (e.g., concept A vs concept B), your confidence should reflect that (~0.55, not 0.85). Reserve high confidence (>0.85) for clear-cut cases.

## Your Task
Label these 4 dimensions:

### Concept (multi-select)
What programming knowledge is needed to produce the response?

Fundamentals:
- control-flow: Conditionals, loops, branching, iteration
- data-types: Primitives, composites, type coercion
- functions: Closures, lambdas, higher-order functions, scope
- data-structures: Arrays, lists, hash tables, trees, graphs, heaps
- object-oriented-programming: Classes, inheritance, polymorphism, encapsulation
- functional-programming: Pure functions, immutability, composition, monads
- recursion: Self-referential calls, base cases, recursive decomposition

Advanced:
- concurrency: Threads, async/await, coroutines, mutex, race conditions, channels
- memory-management: Stack/heap, GC, pointers, references, memory layout
- ownership: Rust ownership, borrow checker, lifetimes, move semantics (Rust-specific)
- type-system: Generics, type inference, traits, interfaces, algebraic types
- error-handling: Exceptions, try/catch, Result/Option patterns, error propagation
- metaprogramming: Reflection, macros, decorators, code generation
- algorithms: Algorithm design and analysis: dynamic programming (DP), greedy, graph theory, geometry, combinatorics, bit manipulation, string algorithms, sorting, searching, divide-and-conquer, backtracking
- iterators: Generators, streams, lazy evaluation, yield

Engineering:
- design-patterns: GoF patterns, SOLID, dependency injection, MVC/MVVM
- architecture: Named architectural patterns and distributed system concepts — microservices, event-driven, CQRS, CAP theorem, saga pattern. NOT basic "how to structure my project" or "separate frontend and backend"
- testing: Unit/integration/e2e testing, TDD, mocking, coverage
- security: Auth, encryption, XSS, CSRF, SQL injection, OAuth, JWT
- database-concepts: Database THEORY — normalization, indexing strategies, ACID/isolation levels, query optimization, sharding. NOT writing a simple CREATE TABLE or basic CRUD queries
- api-protocols: REST, GraphQL, gRPC, WebSocket
- caching: Cache strategies, invalidation, CDN, memoization
- version-control: Git workflows, branching, merge conflicts
- ci-cd: CI/CD pipeline DESIGN and principles, build automation strategies. NOT writing a simple CI config file (→ domain:devops + task:configuration)
- profiling: Understanding profiling tools (cProfile, perf, flame graphs), benchmarking methodology, performance metrics interpretation
- debugging: Systematic debugging methodology (breakpoint debugging, stack trace analysis, bisect/isolation, logging-based diagnosis). NOT the same as intent=debug; tag only when debugging TECHNIQUES are demonstrated or taught

Rules:
- **Cardinality norms**: concept 0-3 (most responses need 0-2), agentic 0-4, constraint 0-2
- Tag 1-3 concepts typically (the CORE knowledge areas), rarely more than 4
- **Empty is a valid answer**: If the response only uses basic syntax, framework APIs, or configuration — concept should be empty []. Not every code sample needs concept tags.
- Choose the umbrella concept, not sub-concepts
- **CRITICAL**: Only use tags from the list above. NEVER invent new concept tags.
- **Framework-specific knowledge is NOT a concept**: React hooks, Vue lifecycle, Django ORM, Flask routing — these are framework APIs, not general programming concepts. Do NOT tag `functions` or `control-flow` just because framework code uses callbacks or conditionals.
- **THRESHOLD for `data-structures`**: Tag ONLY when data structure choice or design is a meaningful focus (e.g., "implement a B-tree", "choose between array vs linked list", "tree traversal", OrderedDict for LRU). Do NOT tag when standard collections (list, dict, array) are merely used as containers.
- **THRESHOLD for `design-patterns`**: Tag ONLY when the query explicitly discusses or implements a NAMED pattern (Factory, Observer, Strategy, DI, SOLID). General OOP structure or code organization does NOT warrant this tag.
- **THRESHOLD for `error-handling`**: Tag ONLY when error handling is a non-trivial focus (custom error types, error propagation strategy, Result/Option). Boilerplate try/catch alone does NOT warrant this tag.
- **THRESHOLD for `algorithms`**: Tag ONLY when algorithm design, complexity analysis, or a specific algorithm (DP, greedy, sorting algorithm) is a meaningful focus. Using a built-in sort() does NOT warrant this tag. Dynamic programming, graph theory, geometry, combinatorics, bit manipulation → all map to `algorithms`.
- **THRESHOLD for `architecture`**: Tag ONLY when the response discusses named architectural patterns (microservices, event-driven, CQRS, saga) or distributed system theory (CAP, consensus, partitioning). "Separate frontend and backend" or "put this in a utils folder" does NOT warrant this tag.
- **THRESHOLD for `database-concepts`**: Tag ONLY when database theory is a meaningful focus (normalization forms, index design tradeoffs, isolation levels, query plan optimization, sharding strategies). Writing a simple CREATE TABLE or basic SELECT queries does NOT warrant this tag.
- **THRESHOLD for `ci-cd`**: Tag ONLY when CI/CD pipeline DESIGN or principles are discussed (pipeline stages, artifact management, deployment strategies like blue-green/canary). Simply writing a GitHub Actions YAML or Dockerfile does NOT warrant this tag — that's task:configuration + domain:devops.
- **THRESHOLD for `concurrency`**: Tag ONLY when concurrency is a meaningful focus — race conditions, synchronization primitives (mutex, semaphore, channel), parallel algorithms, concurrent data structures. Simple `async/await` for sequential I/O (e.g., `await fetch(url)`) does NOT warrant this tag.
- **THRESHOLD for `metaprogramming`**: Tag ONLY when the response involves CREATING custom decorators, metaclasses, macros, or code generation. Simply USING a framework's built-in decorators (e.g., `@app.route`, `@pytest.fixture`, `@dataclass`) does NOT warrant this tag.
- **THRESHOLD for `debugging`**: Tag ONLY when the response demonstrates or teaches debugging TECHNIQUES (systematic isolation, stack trace analysis, using debuggers, bisect strategy). Simply fixing a bug (intent=debug + task=bug-fixing) does NOT warrant this tag. "My sort() broke because of wrong key" → concept=[], NOT concept=debugging.
- **error-handling vs other dimensions**: error-handling = KNOWLEDGE of error patterns (Result/Option, custom errors). Agent retrying on failure → agentic:error-recovery. User asking to add try/catch → task:error-handling-task. "Must handle failures gracefully" → constraint:fault-tolerant.
- **Positive signals** — tag these concepts when you see:
  - Writing custom Python decorators/metaclasses, Rust/C macros, code generation, compiler plugins → `metaprogramming`
  - TTL cache, LRU eviction, memoization, CDN caching → `caching`
  - Concurrent async operations (asyncio.gather, Promise.all), parallel HTTP fetches, thread pools → `concurrency`
  - Class hierarchy with inheritance, encapsulation, polymorphism → `object-oriented-programming`
  - Container security, RBAC, permissions, OAuth/JWT → `security`
  - WebSocket, REST API design, gRPC, JSON-RPC → `api-protocols`
  - Dynamic programming, knapsack, graph BFS/DFS, geometry algorithms, combinatorics → `algorithms`
  - Using pdb/gdb, systematic print-debugging, stack trace walkthrough, bisect isolation → `debugging`
- Examples of correct mapping:
  - mutex/deadlock/race condition → `concurrency`
  - B-tree/hash table/linked list implementation → `data-structures`
  - MVC/SOLID/dependency injection/factory pattern → `design-patterns`
  - microservices/event-driven/CQRS/CAP theorem → `architecture`
  - Python decorator with functools.wraps → `metaprogramming`
  - Webpack/Babel plugin hooking into compiler → `metaprogramming`
  - Cache with TTL + LRU eviction → `caching` + `data-structures`
  - Docker multi-stage build → skip (infrastructure config, not a concept)
  - Flexbox/CSS Grid → skip (too specific, no matching umbrella)
  - React hooks lifecycle → skip (framework-specific, not a concept)
  - K8s YAML configuration → skip (infrastructure config)
  - Package manager comparison → skip (tooling decision, not a concept)
- Examples of INCORRECT tagging:
  - Code uses a list to store results → do NOT tag `data-structures` (incidental usage)
  - Code has a try/catch block → do NOT tag `error-handling` (boilerplate)
  - Code calls sorted() → do NOT tag `algorithms` (trivial usage)
  - Code uses classes → do NOT tag `design-patterns` (basic OOP, not a named pattern)
  - JS var/let/const scoping → do NOT tag `control-flow` or `functions` (basic syntax)
  - React useEffect / Vue watch → do NOT tag `functions` (framework API)
  - Bash for-loop script → do NOT tag `control-flow` (trivial shell scripting)
  - "refactoring", "test creation", "issue analysis" are task labels from Call 1, not concept tags
  - Repo repair trajectories often have concept=[] unless testing/concurrency/error-handling is a real technical focus
  - intent=debug, simple fix like wrong parameter → do NOT tag `debugging` (no debugging methodology demonstrated)

### Agentic (multi-select)
What tool actions and behavioral patterns does the AI agent USE in its response?

Tool Actions (the agent actually invokes these):
- api-calling: Makes HTTP requests to external APIs
- bash-execution: Executes shell commands
- build-execution: Runs build tools/compilers
- code-execution: Runs code snippets for testing
- database-query: Executes database queries
- dependency-installation: Installs packages
- file-operations: Reads, writes, edits, searches files
- git-operations: Git commits, push, branch, etc.
- static-analysis: Runs linters, type checkers
- test-running: Executes test suites
- ui-automation: Browser/GUI automation
- web-search: Searches the web

Behavioral Patterns (cognitive strategies the agent employs):
- context-management: Manages conversation context/memory
- error-recovery: Recovers from errors, retries differently
- iterative-refinement: Improves output through multiple iterations
- multi-file-coordination: Coordinates changes across files
- multi-step-reasoning: Breaks complex problems into steps
- parallel-execution: Runs multiple tasks concurrently
- planning: Creates execution plans before acting
- subagent-management: Delegates to specialized agents
- tool-selection: Chooses the right tool for subtasks
- user-interaction: Asks clarifying questions
- visual-understanding: Understands screenshots/diagrams

Rules:
- Tool Actions: ONLY if the response contains evidence of tool use (tool role messages, command outputs, file read/write)
- Behavioral Patterns: ONLY if the AI agent **actually demonstrates** the pattern in its response
  - `multi-step-reasoning`: Agent explicitly breaks down and executes steps where LATER steps depend on EARLIER step results (NOT just giving a structured explanation or numbered list)
  - `planning`: Agent creates an explicit plan before executing (NOT just answering a planning question)
  - `iterative-refinement`: Agent tries, fails, adjusts, retries (NOT just explaining alternatives)
- **Key distinction**: A thorough explanation is NOT agentic behavior. Q&A without tool use → Agentic should be empty []
- A user asking "how to use Git" does NOT mean the agent uses git-operations
- If there are no tool role messages and no evidence of iterative agent behavior → leave Agentic empty
- **planning vs multi-step-reasoning**: `planning` = agent explicitly LISTS a plan before executing (e.g., "Step 1: ..., Step 2: ..., Let me start with step 1"). `multi-step-reasoning` = agent executes steps where each step's output informs the next (e.g., reads a file → finds the bug → applies fix → tests). A structured explanation with numbered points is NEITHER.
- **preprocessed_signals are heuristic**: Fields in preprocessed_signals (e.g., `has_tool_roles`, `detected_languages`) are automated heuristics and may contain false positives. Always verify against the actual conversation content before tagging.

### Constraint (multi-select)
What non-functional requirements are EXPLICITLY stated or strongly implied?
- accessible: Must follow WCAG accessibility standards
- backward-compatible: Must not break existing APIs/behavior
- deterministic: Same inputs → same outputs
- fault-tolerant: Handle failures gracefully
- gdpr-compliant: EU data protection compliance
- hipaa-compliant: Health data compliance
- idempotent: Repeated execution → same result
- internationalized: Support multiple languages/locales
- lock-free: No mutex/lock primitives allowed
- no-dynamic-allocation: Stack-only memory, no heap
- no-external-dependencies: Standard library only
- no-recursion: Iterative solutions only
- observable: Must include logging/metrics/tracing
- offline-capable: Must work without network connectivity
- pci-dss-compliant: Payment card data compliance
- performance-optimized: Specific performance targets (latency, throughput, memory)
- portable: Cross-platform compatibility
- scalable: Handle growing load
- stateless: No persistent state between calls
- thread-safe: Safe for concurrent multi-thread access
- type-safe: Strict typing, no unsafe casts

Rules:
- ONLY tag constraints explicitly mentioned or strongly implied by the user
- "Make it fast" → performance-optimized; no mention of performance → don't tag
- Most conversations have NO constraints — empty is normal

### Context (single-select)
What is the code scope?

Scope tags (prefer these — based on how much code the response touches):
- snippet: Code fragment, smaller than a function
- single-function: One complete function
- single-file: One complete file with imports
- multi-file: Changes spanning multiple files
- module: Self-contained package/module
- repository: Repo-wide operations

Situational tags (use ONLY when the situation is the defining characteristic):
- greenfield: New project from scratch (explicit "start a new project", scaffold, boilerplate)
- legacy-code: Maintaining/refactoring existing old codebase (explicit "old code", "legacy system", upgrade)
- monorepo: Multiple projects in one repo (explicit monorepo tooling like Nx, Turborepo, Bazel workspaces)
- with-dependencies: Response CENTERS on external package integration (choosing, configuring, troubleshooting a library). NOT just importing a library

Rules:
- Choose ONE that best describes the scope
- **Priority: scope tags first**. Default to a scope tag (snippet → repository). Only use a situational tag when the situation is MORE defining than the code scope
- "New project" → greenfield
- "Refactoring old code" → legacy-code
- A Python script that imports requests → single-file (NOT with-dependencies, the import is incidental)
- "Help me set up a React project with Tailwind" → greenfield (new project is the defining characteristic)
- Scope hierarchy: snippet < single-function < single-file < multi-file < module < repository
  - `snippet`: Code fragment shorter than a complete function (a few lines, an expression)
  - `single-function`: ONE complete function with its logic
  - `single-file`: Complete file with imports, multiple functions/classes
  - `multi-file`: Response creates/modifies files in different paths
- A single SQL statement → snippet
- A complete Python script with imports → single-file
- A GitHub Actions workflow file → single-file (not repository)
- Pure discussion with no actual code → default to `snippet`

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{
  "concept": ["concurrency", "database-concepts"],
  "agentic": ["file-operations", "iterative-refinement"],
  "constraint": ["thread-safe"],
  "context": "multi-file",
  "confidence": {
    "concept": 0.85,
    "agentic": 0.90,
    "constraint": 0.95,
    "context": 0.80
  },
  "unmapped": []
}

Rules for output:
- Use ONLY exact lowercase kebab-case tag IDs from the pools above
- unmapped: [{"dimension":"...","value":"..."}] for missing tags only; use [] if none, with short kebab-case values
- The conversation may contain XML-like tags (<solution>, <tool_call>, etc.), diff markers, or other formatting from the original data source. Ignore all such formatting — focus only on the semantic content."""


CALL2_SYSTEM_COMPACT = """You are an expert SFT data annotator. Label code-related conversations with structured tags across 4 dimensions (capabilities and constraints).

You are given the conversation AND Call 1 results for context.

## Annotation Principles
1. **Evidence-based**: Every tag needs direct evidence. Do not speculate.
2. **Conservative**: Missing a tag < adding a wrong tag.
3. **Label the last turn**: Focus on the last user query and response.
4. **Umbrella concepts**: Use broadest applicable tag (e.g., `concurrency` covers mutex/deadlock).
5. **Agentic = what AI DID**: Not what user asked about. "How to use Git" ≠ `git-operations`.
6. **Call 1 is fallible**: Verify against conversation. Trust conversation over Call 1.
7. **Confidence calibration**: Genuinely torn → ~0.55. Reserve >0.85 for clear-cut cases.

## Dimensions

### Concept (multi-select)
What programming knowledge is needed to produce the response?

Fundamentals:
- control-flow: Conditionals, loops, branching, iteration
- data-types: Primitives, composites, type coercion
- functions: Closures, lambdas, higher-order functions, scope
- data-structures: Arrays, lists, hash tables, trees, graphs, heaps
- object-oriented-programming: Classes, inheritance, polymorphism, encapsulation
- functional-programming: Pure functions, immutability, composition, monads
- recursion: Self-referential calls, base cases, recursive decomposition

Advanced:
- concurrency: Threads, async/await, coroutines, mutex, race conditions, channels
- memory-management: Stack/heap, GC, pointers, references, memory layout
- ownership: Rust ownership, borrow checker, lifetimes, move semantics
- type-system: Generics, type inference, traits, interfaces, algebraic types
- error-handling: Exceptions, try/catch, Result/Option patterns, error propagation
- metaprogramming: Reflection, macros, decorators, code generation
- algorithms: DP, greedy, graph theory, geometry, combinatorics, bit manipulation, string algorithms, sorting, searching, divide-and-conquer, backtracking
- iterators: Generators, streams, lazy evaluation, yield

Engineering:
- design-patterns: GoF patterns, SOLID, dependency injection, MVC/MVVM
- architecture: Named patterns (microservices, event-driven, CQRS, CAP, saga). NOT basic project structure
- testing: Unit/integration/e2e testing, TDD, mocking, coverage
- security: Auth, encryption, XSS, CSRF, SQL injection, OAuth, JWT
- database-concepts: DB theory — normalization, indexing, ACID, query optimization, sharding. NOT simple CRUD
- api-protocols: REST, GraphQL, gRPC, WebSocket
- caching, version-control, ci-cd, profiling, debugging: use only when clearly the primary technical focus

Thresholds — tag ONLY when the concept is a meaningful focus:
| Tag | Tag when... | Skip when... |
|-----|------------|-------------|
| data-structures | DS choice/design is focus: implement B-tree, choose array vs linked list, tree traversal, LRU eviction | Standard list/dict/array as containers — dict for API responses, list for loop results does NOT qualify |
| functions | Closures, higher-order functions, scope chains, function composition are the focus | Code merely defines/calls functions as normal program structure |
| object-oriented-programming | Class hierarchy design, inheritance/polymorphism patterns, encapsulation decisions | Code merely uses classes as containers without OOP design focus |
| data-types | Type conversion challenges, composite type design, type coercion subtleties | Code merely uses standard types (str, int, list) without type-related focus |
| design-patterns | Named pattern discussed (Factory, Observer, SOLID, DI) | General OOP structure |
| error-handling | Non-trivial focus (custom errors, Result/Option, propagation strategy) | Boilerplate try/catch |
| algorithms | Algorithm design/analysis (DP, greedy, graph, geometry) is focus | Using built-in sort(), standard library |
| architecture | Named patterns (microservices, CQRS, CAP, event-driven) | Basic project structure |
| database-concepts | Theory focus (normalization, index design, isolation levels) | Simple CREATE TABLE/CRUD |
| concurrency | Race conditions, sync primitives, parallel algorithms, thread pools | Simple sequential await fetch() |
| metaprogramming | CREATING custom decorators, metaclasses, macros, codegen | USING framework decorators (@app.route, @dataclass) |
| debugging | Debugging TECHNIQUES demonstrated (breakpoints, bisect, stack trace) | Simply fixing a bug (intent=debug, not concept) |
| ci-cd | Pipeline DESIGN and principles | Writing simple CI YAML |

Rules:
- **Cardinality**: 0-3 concepts typical. Rarely more than 4.
- **Empty is valid**: Basic syntax, framework APIs, configuration → concept should be []. Not every code sample needs concept tags.
- **Framework-specific knowledge is NOT a concept**: React hooks, Django ORM, Flask routing are framework APIs, not general programming concepts. Do NOT tag `functions` or `control-flow` just because framework code uses callbacks or conditionals.
- **ONLY use tags from the list above**. NEVER invent new concept tags.
- Common INCORRECT tagging:
  - Code uses list/dict/array as container → NOT `data-structures` (incidental usage)
  - Code has try/catch → NOT `error-handling` (boilerplate)
  - Code calls sorted() → NOT `algorithms` (trivial)
  - Code uses classes → NOT `design-patterns` (basic OOP, not named pattern)
  - Code defines/calls functions normally → NOT `functions` (basic program structure)
  - Agent reads files and applies fixes → NOT `debugging` (agentic behavior)
  - React useEffect / Vue watch → NOT `functions` (framework API)
  - Task-like phrases such as "refactoring", "test creation", "issue analysis" belong to Call 1 task tags, NOT concept tags
  - Repository repair trajectories often have concept=[] unless the response truly hinges on testing/concurrency/error-handling/etc.

### Agentic (multi-select)
What tool actions and behavioral patterns does the AI agent USE in its response?

Tool Actions (requires tool role messages or command output in conversation):
- api-calling: Makes HTTP requests to external APIs
- bash-execution: Executes shell commands (evidence: tool message with `$ command` and output)
- build-execution: Runs build tools/compilers
- code-execution: Runs code in interpreter with output shown (Python REPL, Node.js). NOT compilation (→ build-execution)
- database-query: Executes database queries
- dependency-installation: Installs packages
- file-operations: Reads, writes, edits files via tools
- git-operations: Git commits, push, branch, etc.
- static-analysis: Runs linters, type checkers
- test-running: Executes test suites
- ui-automation, web-search: Browser automation / web search

Behavioral Patterns (demonstrated by agent — NOT just discussed):
- context-management: Manages conversation context/memory
- error-recovery: Recovers from tool errors, retries differently
- iterative-refinement: Try → observe → adjust → retry cycles in tool messages
- multi-file-coordination: Coordinates changes across multiple files
- multi-step-reasoning: Later steps depend on earlier step outputs
- parallel-execution: Runs multiple tasks concurrently
- planning: Creates explicit execution plan before acting
- subagent-management, tool-selection, user-interaction, visual-understanding

Rules:
- NO tool role messages → Tool Actions = []. No tool use + no iterative behavior → Agentic = []
- Q&A without tool calls → Agentic = []. A thorough explanation is NOT agentic behavior
- planning REQUIRES ≥2 tool calls and is RARE (<5%). Agent LISTS a plan then executes with tools (NOT a structured explanation)
- multi-step-reasoning: step N's tool output directly determines step N+1's action (NOT a numbered explanation)
- error-recovery is VERY RARE (<2%). Requires a tool error output + different approach after. User-reported bug ≠ error-recovery
- preprocessed_signals are heuristic. Always verify against actual conversation.

### Constraint (multi-select)
Non-functional requirements EXPLICITLY stated or strongly implied.
- accessible: WCAG accessibility standards
- backward-compatible: Must not break existing APIs/behavior
- deterministic: Same inputs → same outputs
- fault-tolerant: Handle failures gracefully
- gdpr-compliant: EU data protection compliance
- hipaa-compliant: Health data compliance
- idempotent: Repeated execution → same result
- internationalized: Multiple languages/locales
- lock-free: No mutex/lock primitives
- no-dynamic-allocation: Stack-only memory
- no-external-dependencies: Standard library only
- no-recursion: Iterative solutions only
- observable: Must include logging/metrics/tracing
- offline-capable: Must work without network connectivity
- pci-dss-compliant: Payment card data compliance
- performance-optimized: Specific performance targets
- portable: Cross-platform compatibility
- scalable: Handle growing load
- stateless: No persistent state between calls
- thread-safe: Safe for concurrent access
- type-safe: Strict typing, no unsafe casts
Rules: ONLY tag explicitly mentioned constraints. Most conversations → empty.

### Context (single-select)
What is the code scope?

Scope tags (prefer these):
- snippet: Code fragment, smaller than a function (a few lines, an expression, a single SQL statement)
- single-function: ONE complete function with its logic
- single-file: Complete file with imports, multiple functions/classes
- multi-file: Response creates/modifies files in different paths
- module: Self-contained package/module
- repository: Repo-wide operations

Situational tags (use ONLY when the situation is the defining characteristic):
- greenfield: New project from scratch
- legacy-code: Maintaining/refactoring old codebase
- monorepo: Multiple projects in one repo (Nx, Turborepo, Bazel)
- with-dependencies: Response CENTERS on external package integration

Rules:
- Choose ONE. Default to scope tags. Scope hierarchy: snippet < single-function < single-file < multi-file < module < repository
- Q&A with inline code examples (not a complete runnable file) → snippet
- A complete Python script with imports → single-file
- A GitHub Actions workflow file → single-file (not repository)
- Pure discussion with no actual code → snippet
- "New project" → greenfield; "Refactoring old code" → legacy-code
- A Python script that imports requests → single-file (NOT with-dependencies)
- snippet is common (~25% of samples). Use it for code fragments, explanations with examples, algorithm illustrations

## Output Format
Return ONLY valid JSON:
{
  "concept": ["concurrency", "database-concepts"],
  "agentic": ["file-operations", "iterative-refinement"],
  "constraint": ["thread-safe"],
  "context": "multi-file",
  "confidence": {"concept": 0.85, "agentic": 0.90, "constraint": 0.95, "context": 0.80},
  "unmapped": []
}

Use ONLY exact lowercase kebab-case tag IDs from the pools above. unmapped: [{"dimension":"...","value":"..."}] or [].
Ignore XML tags, diff markers, formatting — evaluate semantic content only."""


CALL2_FEWSHOT = [
    # Example 1: Simple Q&A — minimal agentic
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"请解释动态规划的背包问题，0/1 背包和完全背包有什么区别？用 Python 代码举例"},{"from":"gpt","value":"## 动态规划背包问题\\n\\n### 0/1 背包\\n每件物品只能选一次。状态转移：`dp[i][w] = max(dp[i-1][w], dp[i-1][w-wi] + vi)`\\n\\n```python\\ndef knapsack_01(W, weights, values):\\n    n = len(weights)\\n    dp = [[0]*(W+1) for _ in range(n+1)]\\n    for i in range(1, n+1):\\n        for w in range(W+1):\\n            dp[i][w] = dp[i-1][w]\\n            if w >= weights[i-1]:\\n                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])\\n    return dp[n][W]\\n```\\n\\n### 完全背包\\n每件物品可选无限次。只需改一行：`dp[i][w-wi]` 变成 `dp[i][w-wi]`（注意第一维是 i 不是 i-1）\\n\\n```python\\ndef knapsack_complete(W, weights, values):\\n    dp = [0]*(W+1)\\n    for i in range(len(weights)):\\n        for w in range(weights[i], W+1):\\n            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])\\n    return dp[W]\\n```\\n\\n核心区别在于遍历顺序：0/1 背包逆序，完全背包正序。"}]
</conversation>

<call1_result>
{"intent":"learn","language":["python"],"domain":[],"task":["code-explanation"],"difficulty":"intermediate"}
</call1_result>

<preprocessed_signals>
has_tool_roles: false
code_block_count: 2
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"concept":["algorithms","data-structures"],"agentic":[],"constraint":[],"context":"snippet","confidence":{"concept":0.90,"agentic":0.98,"constraint":0.99,"context":0.92},"unmapped":[]}'
    },
    # Example 2: Agentic multi-turn debugging
    # Note: data-structures is correct here because the bug is about map reference
    # semantics (returning internal map vs copy) — this is a data structure design issue,
    # not incidental container usage.
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"我的 Go 程序偶现 panic: concurrent map writes，代码如下..."},{"from":"gpt","value":"让我看看你的代码。问题很可能在 GetAll 方法。"},{"from":"tool","value":"$ cat cache.go\\ntype Cache struct {\\n    mu    sync.Mutex\\n    items map[string]string\\n}\\nfunc (c *Cache) GetAll() map[string]string {\\n    c.mu.Lock()\\n    defer c.mu.Unlock()\\n    return c.items  // BUG: returns reference to internal map\\n}"},{"from":"gpt","value":"找到了！GetAll 返回了 map 的直接引用，调用方可以在没有锁保护的情况下读写这个 map。\\n\\n修复：返回一个拷贝\\n```go\\nfunc (c *Cache) GetAll() map[string]string {\\n    c.mu.Lock()\\n    defer c.mu.Unlock()\\n    copy := make(map[string]string, len(c.items))\\n    for k, v := range c.items {\\n        copy[k] = v\\n    }\\n    return copy\\n}\\n```\\n\\n或者考虑使用 sync.RWMutex 来允许并发读。"}]
</conversation>

<call1_result>
{"intent":"debug","language":["go"],"domain":[],"task":["bug-fixing"],"difficulty":"advanced"}
</call1_result>

<preprocessed_signals>
has_tool_roles: true
tool_names: ["bash"]
code_block_count: 1
total_turns: 4
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"concept":["concurrency","data-structures"],"agentic":["file-operations","multi-step-reasoning"],"constraint":["thread-safe"],"context":"single-file","confidence":{"concept":0.92,"agentic":0.88,"constraint":0.85,"context":0.90},"unmapped":[]}'
    },
    # Example 3: Complex agentic build
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"帮我配置 GitHub Actions CI：push 时跑 pytest，通过后构建 Docker 镜像推到 ECR"},{"from":"gpt","value":"好的，我来创建 CI 配置。先看下项目结构。"},{"from":"tool","value":"$ ls\\nDockerfile  requirements.txt  src/  tests/  pyproject.toml"},{"from":"gpt","value":"项目结构清楚了。我来创建 workflow 文件。"},{"from":"tool","value":"$ cat > .github/workflows/ci.yml << 'EOF'\\nname: CI/CD\\non:\\n  push:\\n    branches: [main]\\njobs:\\n  test:\\n    runs-on: ubuntu-latest\\n    steps:\\n      - uses: actions/checkout@v4\\n      - uses: actions/setup-python@v5\\n        with: {python-version: '3.11'}\\n      - run: pip install -r requirements.txt\\n      - run: pytest tests/ -v --tb=short\\n  build-push:\\n    needs: test\\n    runs-on: ubuntu-latest\\n    steps:\\n      - uses: actions/checkout@v4\\n      - uses: aws-actions/configure-aws-credentials@v4\\n        with:\\n          role-to-assume: ${{ secrets.AWS_ROLE }}\\n          aws-region: us-east-1\\n      - uses: aws-actions/amazon-ecr-login@v2\\n      - run: |\\n          docker build -t $ECR_REPO:$GITHUB_SHA .\\n          docker push $ECR_REPO:$GITHUB_SHA\\nEOF"},{"from":"gpt","value":"CI/CD pipeline 已创建。流程：\\n1. push 到 main → 自动跑 pytest\\n2. 测试通过 → 构建 Docker 镜像\\n3. 推送到 ECR（用 OIDC 认证，无需 access key）\\n\\n你需要在 repo settings 里配置 `AWS_ROLE` secret。"}]
</conversation>

<call1_result>
{"intent":"build","language":["yaml","python","dockerfile"],"domain":["devops","cloud-computing"],"task":["configuration","deployment"],"difficulty":"intermediate"}
</call1_result>

<preprocessed_signals>
has_tool_roles: true
tool_names: ["bash"]
code_block_count: 0
total_turns: 6
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"concept":[],"agentic":["file-operations","bash-execution","multi-file-coordination","planning"],"constraint":[],"context":"repository","confidence":{"concept":0.92,"agentic":0.92,"constraint":0.97,"context":0.88},"unmapped":[]}'
    },
    # Example 4: Standard framework CRUD — empty concept
    # Key calibration: routine framework usage does NOT need concept tags.
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"帮我用 Django REST Framework 写一个 Product 的 ViewSet，支持 CRUD，加上分页和权限控制"},{"from":"gpt","value":"好的，这是一个标准的 DRF ViewSet 实现：\\n\\n```python\\nfrom rest_framework import viewsets, permissions, pagination\\nfrom .models import Product\\nfrom .serializers import ProductSerializer\\n\\nclass StandardPagination(pagination.PageNumberPagination):\\n    page_size = 20\\n    page_size_query_param = 'page_size'\\n    max_page_size = 100\\n\\nclass ProductViewSet(viewsets.ModelViewSet):\\n    queryset = Product.objects.all()\\n    serializer_class = ProductSerializer\\n    pagination_class = StandardPagination\\n    permission_classes = [permissions.IsAuthenticatedOrReadOnly]\\n\\n    def perform_create(self, serializer):\\n        serializer.save(created_by=self.request.user)\\n```\\n\\n然后在 urls.py 注册：\\n```python\\nfrom rest_framework.routers import DefaultRouter\\nfrom .views import ProductViewSet\\n\\nrouter = DefaultRouter()\\nrouter.register('products', ProductViewSet)\\nurlpatterns = router.urls\\n```\\n\\n这样就有了完整的 CRUD + 分页 + 权限。GET 所有人可访问，POST/PUT/DELETE 需要登录。"}]
</conversation>

<call1_result>
{"intent":"build","language":["python"],"domain":["web-backend","api-development"],"task":["feature-implementation","api-design"],"difficulty":"intermediate"}
</call1_result>

<preprocessed_signals>
has_tool_roles: false
code_block_count: 2
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"concept":[],"agentic":[],"constraint":[],"context":"single-file","confidence":{"concept":0.90,"agentic":0.98,"constraint":0.97,"context":0.85},"unmapped":[]}'
    },
    # Example 4 (compact-only): Compact bash-execution anchor — shows bash+file-ops+multi-step WITHOUT planning or error-recovery
    # Key calibration: running tests is bash-execution. Reading file then fixing based on what was found = multi-step-reasoning.
    # No planning (agent doesn't list steps upfront). No error-recovery (test failure was pre-existing, not a mid-task failure).
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"我的 Go 测试失败了，帮我看看"},{"from":"gpt","value":"好的，我看一下测试结果"},{"from":"tool","value":"$ go test ./... -run TestUserService\\nFAIL\\n--- FAIL: TestUserService/TestCreate (0.01s)\\n    service_test.go:45: expected user ID 1, got 0"},{"from":"gpt","value":"看到了，让我检查 Create 方法的实现"},{"from":"tool","value":"$ cat service.go\\nfunc (s *UserService) Create(user User) (int, error) {\\n    result, err := s.db.Exec(\\"INSERT INTO users (name) VALUES (?)\\", user.Name)\\n    if err != nil { return 0, err }\\n    id, _ := result.LastInsertId()\\n    return int(id), nil\\n}"},{"from":"gpt","value":"找到了。`result.LastInsertId()` 返回 `(int64, error)`，这里忽略了 error 导致 id 为 0。\\n\\n修复：\\n```go\\nfunc (s *UserService) Create(user User) (int, error) {\\n    result, err := s.db.Exec(\\"INSERT INTO users (name) VALUES (?)\\", user.Name)\\n    if err != nil { return 0, err }\\n    id, err := result.LastInsertId()\\n    if err != nil { return 0, err }\\n    return int(id), nil\\n}\\n```"}]
</conversation>

<call1_result>
{"intent":"debug","language":["go"],"domain":[],"task":["bug-fixing"],"difficulty":"intermediate"}
</call1_result>

<preprocessed_signals>
has_tool_roles: true
tool_names: ["bash"]
code_block_count: 1
total_turns: 6
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        # bash-execution: agent ran `go test` and `cat` (shell commands with output)
        # file-operations: agent read service.go
        # multi-step-reasoning: test output revealed failure → reading file revealed root cause → fix targeted that specific finding
        # NOT planning: no upfront plan was listed before execution
        # NOT error-recovery: the test failure was pre-existing (reported by user), not a mid-task failure the agent encountered
        "content": '{"concept":["error-handling"],"agentic":["bash-execution","file-operations","multi-step-reasoning"],"constraint":[],"context":"single-file","confidence":{"concept":0.85,"agentic":0.88,"constraint":0.98,"context":0.90},"unmapped":[]}'
    },
    # Example 6: SWE-style repository repair — conservative concept, rich agentic
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"仓库里的用户创建接口回归了，跑 `pytest tests/test_users.py -q` 会挂。请检查相关文件，修好实现，如果需要也一起更新测试。"},{"from":"gpt","value":"我先运行失败用例，再定位实现。"},{"from":"tool","value":"$ pytest tests/test_users.py -q\\nFAILED tests/test_users.py::test_create_user_returns_id\\nE assert 0 == 1"},{"from":"gpt","value":"我去看 service 层代码。"},{"from":"tool","value":"$ sed -n '1,120p' app/services/user_service.py\\ndef create_user(repo, payload):\\n    result = repo.insert(payload)\\n    return 0 if result is None else result.id"},{"from":"gpt","value":"已经定位到问题：成功插入时应该返回 `result.id`，失败时不能静默返回 0。修复如下，并保留测试断言验证真实 id。\\n\\n```python\\ndef create_user(repo, payload):\\n    result = repo.insert(payload)\\n    if result is None:\\n        raise ValueError('insert failed')\\n    return result.id\\n```"}]
</conversation>

<call1_result>
{"intent":"debug","language":["python"],"domain":[],"task":["code-exploration","bug-fixing","testing-task"],"difficulty":"upper-intermediate"}
</call1_result>

<preprocessed_signals>
has_tool_roles: true
tool_names: ["bash"]
code_block_count: 1
total_turns: 6
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"concept":[],"agentic":["bash-execution","file-operations","multi-step-reasoning"],"constraint":[],"context":"repository","confidence":{"concept":0.91,"agentic":0.91,"constraint":0.98,"context":0.87},"unmapped":[]}'
    },
]


# ─── Compact few-shot subsets (for --prompt-mode compact) ──

# Call 1: short anchors with balanced difficulty coverage to avoid intermediate collapse
CALL1_FEWSHOT_COMPACT = [
    # Beginner
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"代码输出什么？\\n```python\\nnums=[1,2,3]\\nprint([x*2 for x in nums if x%2])\\n```"},{"from":"gpt","value":"输出 `[2,6]`。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["python"]
has_tool_roles: false
code_block_count: 1
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"learn","language":["python"],"domain":[],"task":["code-explanation"],"difficulty":"beginner","confidence":{"intent":0.96,"language":0.99,"domain":0.95,"task":0.95,"difficulty":0.92},"unmapped":[]}'
    },
    # Intermediate
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"用 Flask 写一个用户 CRUD API，用 SQLite 存储"},{"from":"gpt","value":"可以用 Flask + sqlite3 实现 `/users` 的 GET/POST/PUT/DELETE，并返回 JSON。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["python"]
has_tool_roles: false
code_block_count: 0
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"build","language":["python","sql"],"domain":["web-backend","api-development"],"task":["feature-implementation","api-design"],"difficulty":"intermediate","confidence":{"intent":0.98,"language":0.94,"domain":0.94,"task":0.90,"difficulty":0.88},"unmapped":[]}'
    },
    # Upper-intermediate
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"把这个 Node.js callback 路由改成 async/await，要求保留 redis 失败时回退 rows.length"},{"from":"gpt","value":"可以改为外层 try/catch + 内层 redis try/catch，保持原有降级逻辑不变。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["javascript"]
has_tool_roles: false
code_block_count: 0
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"modify","language":["javascript"],"domain":["web-backend"],"task":["code-refactoring","error-handling-task"],"difficulty":"upper-intermediate","confidence":{"intent":0.97,"language":0.99,"domain":0.88,"task":0.92,"difficulty":0.86},"unmapped":[]}'
    },
    # Advanced
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"实现分布式限流：Redis 滑动窗口，按用户/IP，Redis 故障时降级本地并保证线程安全"},{"from":"gpt","value":"可以用 Redis ZSET 维护窗口计数，并在连接异常时回退到加锁的本地计数器。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["python"]
has_tool_roles: false
code_block_count: 0
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"build","language":["python"],"domain":["web-backend"],"task":["feature-implementation"],"difficulty":"advanced","confidence":{"intent":0.98,"language":0.99,"domain":0.86,"task":0.94,"difficulty":0.90},"unmapped":[]}'
    },
    # Expert
    {
        "role": "user",
        "content": """<conversation>
[{"from":"human","value":"用 Rust 设计一个无锁 MPMC 队列，解释 ABA 风险与内存序选择"},{"from":"gpt","value":"需要基于原子 CAS 设计队列节点推进策略，并结合 hazard pointers/epoch 回收处理 ABA 与内存可见性。"}]
</conversation>

<preprocessed_signals>
detected_languages: ["rust"]
has_tool_roles: false
code_block_count: 0
total_turns: 2
</preprocessed_signals>"""
    },
    {
        "role": "assistant",
        "content": '{"intent":"build","language":["rust"],"domain":["systems-programming"],"task":["feature-implementation"],"difficulty":"expert","confidence":{"intent":0.96,"language":0.99,"domain":0.90,"task":0.90,"difficulty":0.92},"unmapped":[]}'
    },
]

# Call 2: keep 3 of 5 — custom bash/debug(no-planning anchor), Q&A knapsack(empty agentic), empty CRUD(anchor)
# Balance: 1/3 agentic, 2/3 empty — prevents multi-step-reasoning over-tagging
# Intentionally avoids fewshot examples that show planning=true or error-recovery=true
CALL2_FEWSHOT_COMPACT = CALL2_FEWSHOT[8:10] + CALL2_FEWSHOT[0:2] + CALL2_FEWSHOT[6:8]


def build_call1_messages(conversation_json, preprocessed_signals, compact=False):
    """Build messages for Call 1 labeling."""
    user_content = f"""<conversation>
{conversation_json}
</conversation>

<preprocessed_signals>
{preprocessed_signals}
</preprocessed_signals>"""

    fewshot = CALL1_FEWSHOT_COMPACT if compact else CALL1_FEWSHOT
    system = CALL1_SYSTEM_COMPACT if compact else CALL1_SYSTEM
    messages = [{"role": "system", "content": system}]
    messages.extend(fewshot)
    messages.append({"role": "user", "content": user_content})
    return messages


def build_call2_messages(conversation_json, preprocessed_signals, call1_result, compact=False):
    """Build messages for Call 2 labeling, including Call 1 results as context."""
    import json
    call1_str = json.dumps(call1_result, ensure_ascii=False) if isinstance(call1_result, dict) else str(call1_result)

    user_content = f"""<conversation>
{conversation_json}
</conversation>

<call1_result>
{call1_str}
</call1_result>

<preprocessed_signals>
{preprocessed_signals}
</preprocessed_signals>"""

    fewshot = CALL2_FEWSHOT_COMPACT if compact else CALL2_FEWSHOT
    system = CALL2_SYSTEM_COMPACT if compact else CALL2_SYSTEM
    messages = [{"role": "system", "content": system}]
    messages.extend(fewshot)
    messages.append({"role": "user", "content": user_content})
    return messages


# ─────────────────────────────────────────────────────────
# Tag pools for validation
# ─────────────────────────────────────────────────────────

TAG_POOLS = {
    "intent": {"learn", "build", "modify", "debug", "review", "decide"},
    "difficulty": {"beginner", "intermediate", "upper-intermediate", "advanced", "expert"},
    "context": {"snippet", "single-function", "single-file", "multi-file", "module",
                "repository", "monorepo", "greenfield", "legacy-code", "with-dependencies"},
    "language": {
        "ada", "actionscript", "apl", "applescript", "arkts", "ascendc", "assembly",
        "autohotkey", "bazel", "c", "clojure",
        "cmake", "cobol", "cpp", "crystal", "csharp", "css", "dart", "delphi", "dockerfile",
        "dotenv", "ejs", "elixir", "erb", "erlang", "fortran", "fsharp", "gdscript", "glsl",
        "go", "gradle", "groovy", "gml", "handlebars", "haskell", "hcl", "hlsl", "html",
        "ini", "java", "javascript", "jinja", "json", "julia", "kotlin", "latex", "liquid",
        "lisp", "lua", "mql5", "makefile", "markdown", "mathematica", "matlab", "maven",
        "maxscript", "nginx-config", "nim", "nix", "objective-c", "ocaml", "pascal", "perl",
        "php", "pinescript", "powershell", "prolog", "properties", "python", "r",
        "racket", "restructuredtext", "ruby", "rust", "scala", "scheme", "shell", "smalltalk",
        "solidity", "sql", "swift", "toml", "typescript", "vba", "verilog", "vhdl", "vyper",
        "xml", "yaml", "zig"
    },
    "domain": {
        "api-development", "automation", "bioinformatics", "blockchain", "cli-tool",
        "cloud-computing", "compiler-development", "competitive-programming", "compliance", "computer-vision",
        "cybersecurity", "data-engineering", "data-science", "database-administration",
        "desktop-application", "devops", "e-commerce", "embedded-systems",
        "financial-technology", "game-development", "geospatial", "graphics-and-xr",
        "healthcare-technology", "accessibility", "internationalization", "iot",
        "machine-learning", "media-processing", "mobile-development",
        "natural-language-processing", "network-programming", "operating-systems",
        "real-time-systems", "robotics", "scientific-computing", "search-engineering",
        "systems-programming", "web-backend", "web-frontend"
    },
    "concept": {
        "control-flow", "data-types", "functions", "data-structures",
        "object-oriented-programming", "functional-programming", "recursion",
        "concurrency", "memory-management", "ownership", "type-system", "error-handling",
        "metaprogramming", "algorithms", "iterators", "design-patterns", "architecture",
        "testing", "security", "database-concepts", "api-protocols", "caching",
        "version-control", "ci-cd", "profiling", "debugging"
    },
    "task": {
        "api-design", "bug-fixing", "code-completion", "code-explanation", "code-exploration",
        "code-optimization", "code-refactoring", "code-review-task", "code-translation",
        "configuration", "dependency-management", "deployment", "documentation",
        "error-handling-task", "feature-implementation", "logging", "migration",
        "monitoring", "performance-analysis", "schema-design", "security-audit",
        "testing-task"
    },
    "agentic": {
        "api-calling", "bash-execution", "build-execution", "code-execution",
        "database-query", "dependency-installation", "file-operations", "git-operations",
        "static-analysis", "test-running", "ui-automation", "web-search",
        "context-management", "error-recovery", "iterative-refinement",
        "multi-file-coordination", "multi-step-reasoning", "parallel-execution",
        "planning", "subagent-management", "tool-selection", "user-interaction",
        "visual-understanding"
    },
    "constraint": {
        "accessible", "backward-compatible", "deterministic", "fault-tolerant",
        "gdpr-compliant", "hipaa-compliant", "idempotent", "internationalized",
        "lock-free", "no-dynamic-allocation", "no-external-dependencies", "no-recursion",
        "observable", "offline-capable", "pci-dss-compliant", "performance-optimized",
        "portable", "scalable", "stateless", "thread-safe", "type-safe"
    },
}

SINGLE_SELECT = {"intent", "difficulty", "context"}
MULTI_SELECT = {"language", "domain", "concept", "task", "agentic", "constraint"}
