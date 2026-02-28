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
- `learn`: Understand a concept/technique. Signals: "explain", "how does X work", "difference between"
- `build`: Create new functionality. Signals: "help me write", "implement", "create", "build"
- `debug`: Find and fix a problem. Signals: "error", "doesn't work", "why does it fail", "bug"
- `review`: Get feedback on existing code. Signals: "review this", "any issues", "improvements"
- `decide`: Choose between options. Signals: "A vs B", "which should I", "how to choose"

### Language (multi-select)
What programming languages appear in the conversation?
Common: python, javascript, typescript, java, go, rust, c, cpp, csharp, ruby, php, swift, kotlin, sql, html, css, shell, dockerfile, yaml, json, markdown, hcl, xml, toml
Other: ada, apl, assembly, bazel, clojure, cmake, cobol, crystal, dart, dotenv, elixir, erb, erlang, fortran, fsharp, gradle, groovy, handlebars, haskell, ini, jinja, julia, latex, liquid, lisp, lua, makefile, matlab, maven, nginx-config, nim, objective-c, ocaml, perl, powershell, prolog, properties, r, racket, restructuredtext, ruby, scala, scheme, smalltalk, solidity, verilog, vhdl, vyper, zig
Rules:
- Detect from code blocks, framework mentions (Django→python, React→typescript, Spring Boot→java)
- Config file formats count (Docker Compose→yaml, Terraform→hcl)
- If no specific language is involved (pure architecture discussion), leave empty

### Domain (multi-select)
What application area does this belong to?
- api-development: Building APIs
- automation: Automating tasks/workflows
- blockchain: Smart contracts, DeFi, Web3
- cli-tool: Command-line tools
- cloud-computing: Cloud platforms, serverless, edge
- compiler-development: Compilers, interpreters, language tooling
- cybersecurity: Security engineering as an APPLICATION DOMAIN (pentesting, threat modeling, vulnerability research, appsec). For security KNOWLEDGE (auth, encryption, XSS), use concept:security instead
- data-engineering: Data pipelines, ETL, warehouses
- data-science: Statistical analysis, analytics
- database-administration: Database management/optimization
- desktop-application: Native desktop apps
- devops: CI/CD, SRE, platform engineering, observability
- e-commerce: Online shopping, payment systems
- embedded-systems: Firmware, hardware programming
- financial-technology: Fintech applications
- game-development: Video games, interactive entertainment
- graphics-and-xr: 3D graphics, AR/VR, shaders
- healthcare-technology: Medical/health applications
- machine-learning: ML, deep learning, neural nets, MLOps
- media-processing: Audio/video processing
- mobile-development: Mobile apps (iOS, Android)
- natural-language-processing: NLP, text processing
- network-programming: Network/socket programming
- operating-systems: OS/kernel development
- real-time-systems: RTOS, strict timing constraints
- scientific-computing: HPC, numerical methods, simulation
- search-engineering: Search systems, information retrieval
- systems-programming: Low-level system software
- web-backend: Server-side web development
- web-frontend: Client-side web development
- (also: accessibility, bioinformatics, compliance, computer-vision, geospatial, internationalization, iot, robotics)
Rules:
- Tag the application SCENARIO, not the technology
- A query can span multiple domains (e.g., "SaaS payment" → web-backend + e-commerce + financial-technology)
- Pure algorithm practice with no application context → empty
- Prefer the most specific domain tag
- "algorithm", "data-structure", "documentation" are NOT domains. Algorithm/DS practice → concept:algorithms or concept:data-structures, domain should be empty. Writing docs → task:documentation, not a domain.

### Task (multi-select)
What type of work is being done?
- api-design: Define endpoints, request/response formats
- bug-fixing: Fix bugs, identify root cause
- code-completion: Complete partial code
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
- error-handling-task: Add error handling/validation
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

### Difficulty (single-select)
What coding ability level is needed to produce a good response?
- `beginner`: Basic syntax, simple API calls, standard library usage (e.g., Python list operations, HTML/CSS basics)
- `intermediate`: Framework usage, common design patterns, multi-component coordination (e.g., Flask REST API, React components)
- `advanced`: Performance optimization, complex architecture, deep debugging, multi-system integration (e.g., distributed rate limiter, connection pool implementation)
- `expert`: Deep internals knowledge, cutting-edge techniques, large-scale system design (e.g., Rust Pin/Unpin mechanism, Linux kernel scheduler, lock-free data structures, JIT compiler)

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
- unmapped: any tags you wanted to assign but couldn't find in the pool (free text)"""


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
        "content": '{"intent":"build","language":["typescript","css","json"],"domain":["web-frontend"],"task":["configuration","feature-implementation"],"difficulty":"beginner","confidence":{"intent":0.98,"language":0.90,"domain":0.95,"task":0.85,"difficulty":0.90},"unmapped":[]}'
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
3. **Label the last turn**: Focus on the last user query and its response.
4. **Umbrella concepts**: Use the broadest applicable concept tag. E.g., use `concurrency` (not "mutex" or "deadlock" separately — those are sub-concepts captured by the umbrella).
5. **Agentic = what the AI agent DID**: Not what the user asked about. "How to use Git" ≠ `git-operations`. Only tag if the response actually performs the action.

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
- architecture: Microservices, event-driven, CQRS, distributed systems, CAP theorem
- testing: Unit/integration/e2e testing, TDD, mocking, coverage
- security: Auth, encryption, XSS, CSRF, SQL injection, OAuth, JWT
- database-concepts: Schema design, normalization, indexing, ACID, transactions, query optimization
- api-protocols: REST, GraphQL, gRPC, WebSocket
- caching: Cache strategies, invalidation, CDN, memoization
- version-control: Git workflows, branching, merge conflicts
- ci-cd: CI/CD pipelines, build automation, DevOps practices
- profiling: Understanding profiling tools (cProfile, perf, flame graphs), benchmarking methodology, performance metrics interpretation
- debugging: Systematic debugging methodology (breakpoint debugging, stack trace analysis, bisect/isolation, logging-based diagnosis). NOT the same as intent=debug; tag only when debugging TECHNIQUES are demonstrated or taught

Rules:
- Tag 1-3 concepts typically (the CORE knowledge areas), rarely more than 4
- **Empty is a valid answer**: If the response only uses basic syntax, framework APIs, or configuration — concept should be empty []. Not every code sample needs concept tags.
- Choose the umbrella concept, not sub-concepts
- **CRITICAL**: Only use tags from the list above. NEVER invent new concept tags.
- **Framework-specific knowledge is NOT a concept**: React hooks, Vue lifecycle, Django ORM, Flask routing — these are framework APIs, not general programming concepts. Do NOT tag `functions` or `control-flow` just because framework code uses callbacks or conditionals.
- **THRESHOLD for `data-structures`**: Tag ONLY when data structure choice or design is a meaningful focus (e.g., "implement a B-tree", "choose between array vs linked list", "tree traversal", OrderedDict for LRU). Do NOT tag when standard collections (list, dict, array) are merely used as containers.
- **THRESHOLD for `design-patterns`**: Tag ONLY when the query explicitly discusses or implements a NAMED pattern (Factory, Observer, Strategy, DI, SOLID). General OOP structure or code organization does NOT warrant this tag.
- **THRESHOLD for `error-handling`**: Tag ONLY when error handling is a non-trivial focus (custom error types, error propagation strategy, Result/Option). Boilerplate try/catch alone does NOT warrant this tag.
- **THRESHOLD for `algorithms`**: Tag ONLY when algorithm design, complexity analysis, or a specific algorithm (DP, greedy, sorting algorithm) is a meaningful focus. Using a built-in sort() does NOT warrant this tag. Dynamic programming, graph theory, geometry, combinatorics, bit manipulation → all map to `algorithms`.
- **THRESHOLD for `debugging`**: Tag ONLY when the response demonstrates or teaches debugging TECHNIQUES (systematic isolation, stack trace analysis, using debuggers, bisect strategy). Simply fixing a bug (intent=debug + task=bug-fixing) does NOT warrant this tag. "My sort() broke because of wrong key" → concept=[], NOT concept=debugging.
- **error-handling vs other dimensions**: error-handling = KNOWLEDGE of error patterns (Result/Option, custom errors). Agent retrying on failure → agentic:error-recovery. User asking to add try/catch → task:error-handling-task. "Must handle failures gracefully" → constraint:fault-tolerant.
- **Positive signals** — tag these concepts when you see:
  - Python decorators, macros, code generation, compiler plugins → `metaprogramming`
  - TTL cache, LRU eviction, memoization, CDN caching → `caching`
  - async/await, Promise chains, concurrent HTTP fetches → `concurrency`
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
  - `multi-step-reasoning`: Agent explicitly breaks down and executes steps (NOT just giving a detailed explanation)
  - `planning`: Agent creates an explicit plan before executing (NOT just answering a planning question)
  - `iterative-refinement`: Agent tries, fails, adjusts, retries (NOT just explaining alternatives)
- **Key distinction**: A thorough explanation is NOT agentic behavior. Q&A without tool use → Agentic should be empty []
- A user asking "how to use Git" does NOT mean the agent uses git-operations
- If there are no tool role messages and no evidence of iterative agent behavior → leave Agentic empty

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
- snippet: Code fragment, smaller than a function
- single-function: One complete function
- single-file: One complete file with imports
- multi-file: Changes spanning multiple files
- module: Self-contained package/module
- repository: Repo-wide operations
- monorepo: Multiple projects in one repo
- greenfield: New project from scratch
- legacy-code: Maintaining/refactoring existing code
- with-dependencies: Uses external packages/libraries

Rules:
- Choose ONE that best describes the scope
- "New project" → greenfield
- "Refactoring old code" → legacy-code
- Scope hierarchy: snippet < single-function < single-file < multi-file < module < repository
  - `snippet`: Code fragment shorter than a complete function (a few lines, an expression)
  - `single-function`: ONE complete function with its logic
  - `single-file`: Complete file with imports, multiple functions/classes
  - `multi-file`: Response creates/modifies files in different paths
- A single SQL statement → snippet
- A complete Python script with imports → single-file
- A GitHub Actions workflow file → single-file (not repository)

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
}"""


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
        "content": '{"concept":["ci-cd","testing"],"agentic":["file-operations","bash-execution","multi-file-coordination","planning"],"constraint":[],"context":"repository","confidence":{"concept":0.90,"agentic":0.92,"constraint":0.97,"context":0.88},"unmapped":[]}'
    },
]


def build_call1_messages(conversation_json, preprocessed_signals):
    """Build messages for Call 1 labeling."""
    user_content = f"""<conversation>
{conversation_json}
</conversation>

<preprocessed_signals>
{preprocessed_signals}
</preprocessed_signals>"""

    messages = [{"role": "system", "content": CALL1_SYSTEM}]
    messages.extend(CALL1_FEWSHOT)
    messages.append({"role": "user", "content": user_content})
    return messages


def build_call2_messages(conversation_json, preprocessed_signals, call1_result):
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

    messages = [{"role": "system", "content": CALL2_SYSTEM}]
    messages.extend(CALL2_FEWSHOT)
    messages.append({"role": "user", "content": user_content})
    return messages


# ─────────────────────────────────────────────────────────
# Tag pools for validation
# ─────────────────────────────────────────────────────────

TAG_POOLS = {
    "intent": {"learn", "build", "debug", "review", "decide"},
    "difficulty": {"beginner", "intermediate", "advanced", "expert"},
    "context": {"snippet", "single-function", "single-file", "multi-file", "module",
                "repository", "monorepo", "greenfield", "legacy-code", "with-dependencies"},
    "language": {
        "ada", "apl", "arkts", "ascendc", "assembly", "bazel", "c", "clojure", "cmake",
        "cobol", "cpp", "crystal", "csharp", "css", "dart", "dockerfile", "dotenv", "ejs",
        "elixir", "erb", "erlang", "fortran", "fsharp", "go", "gradle", "groovy",
        "handlebars", "haskell", "hcl", "html", "ini", "java", "javascript", "jinja",
        "json", "julia", "kotlin", "latex", "liquid", "lisp", "lua", "makefile", "markdown",
        "matlab", "maven", "nginx-config", "nim", "objective-c", "ocaml", "perl", "php",
        "powershell", "prolog", "properties", "python", "r", "racket", "restructuredtext",
        "ruby", "rust", "scala", "scheme", "shell", "smalltalk", "solidity", "sql", "swift",
        "toml", "typescript", "verilog", "vhdl", "vyper", "xml", "yaml", "zig"
    },
    "domain": {
        "api-development", "automation", "bioinformatics", "blockchain", "cli-tool",
        "cloud-computing", "compiler-development", "compliance", "computer-vision",
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
        "observable", "pci-dss-compliant", "performance-optimized", "portable",
        "scalable", "stateless", "thread-safe", "type-safe"
    },
}

SINGLE_SELECT = {"intent", "difficulty", "context"}
MULTI_SELECT = {"language", "domain", "concept", "task", "agentic", "constraint"}
