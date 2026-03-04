"""
Value Scoring Prompts (Pass 2)

System prompt, few-shot examples, and message builders for the LLM-based
value assessment of SFT training data.
"""

import json

# ─────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────

SCORING_SYSTEM = """\
You are an expert evaluator of SFT (Supervised Fine-Tuning) training data for code-generation LLMs.

Your task: assess the training VALUE of a single conversation sample across three dimensions.

## Dimension 1: Complexity (1-10)

Rate the technical complexity of the task. Provide sub-scores and an overall:

- **instruction** (1-10): How challenging is the question/request itself?
- **analytical_depth** (1-10): How many logical steps or how deep the reasoning needed?
- **implementation** (1-10): How difficult is the actual code implementation?
- **overall** (1-10): Holistic complexity assessment.

Score anchors:
  1-2: Trivial — print("hello"), variable assignment, basic syntax questions
  3-4: Basic — simple CRUD, basic data manipulation, single-function tasks, straightforward API/library usage
  5-6: Intermediate — standard design patterns, moderate algorithms (sorting, BFS/DFS), multi-step logic, typical web/backend features
  7-8: Advanced — system design, concurrent programming, complex algorithms (DP, graph theory), multi-module architecture, performance-critical optimization
  9-10: Expert — novel algorithm design, compiler/interpreter implementation, kernel-level programming, cutting-edge techniques

Calibration:
- Most coding tasks fall in the 3-6 range. A standard Flask/Django endpoint is 4, not 7.
- 7+ requires genuinely advanced technical depth (concurrency, distributed systems, complex algorithm design).
- The difficulty tag in <meta> is a rough reference only (it may be inaccurate). Always assess complexity independently based on the actual task content.
- Reserve 7-8 for tasks requiring specialized knowledge most mid-level developers lack.
- For agentic/tool-use tasks, assess the orchestration complexity: how many tools, how complex the decision logic for choosing tools, and how well errors are handled across tool calls.
- Agentic complexity granularity — use the full scale, avoid locking into just 2-3 values:
  3-4: Simple tool calls with linear flow (read file → apply fix → done)
  5: Multi-step tool use with straightforward decision logic
  6: Moderate orchestration — branching based on tool output, basic error recovery
  7: Complex multi-tool coordination — iterative debugging loops, multiple files, conditional branching
  8: Advanced orchestration — cross-file reasoning, hypothesis testing via tools, backtracking on failed approaches
  9-10: Novel agent strategies — creative tool composition, complex search through solution space

## Dimension 2: Quality (1-10)

Rate the quality of the response. Provide sub-scores and an overall:

- **correctness** (1-10): Is the code/solution functionally correct? This is the most important sub-score.
- **code_quality** (1-10): Code style, naming, structure, best practices, maintainability.
- **explanation** (1-10): Clarity and educational value of explanations.
- **completeness** (1-10): Does the response fully address the instruction?
- **overall** (1-10): Holistic quality assessment.

Score anchors:
  1-2: Severely flawed — contains critical bugs, completely wrong approach
  3-4: Below average — partially correct but missing key aspects, poor code style
  5-6: Acceptable — mostly correct, adequate explanation, standard code quality
  7-8: Good — correct, well-structured code, clear explanation, handles edge cases
  9-10: Excellent — perfectly correct, production-quality code, insightful explanation, comprehensive

Calibration:
- A response with bugs in core logic is 3-4 at most for correctness, regardless of code style or explanation quality.
- "Mostly works but misses edge cases" is 5-6, not 7-8.
- overall should never exceed correctness by more than 2 points. Buggy code cannot be "high quality".
- Reserve 9-10 for code you would merge into production without changes. 9-10 is EXCEPTIONAL — requires production-ready code with excellent error handling, comprehensive edge cases, and insightful explanations.
- Expected distribution: ~15% at 1-3, ~50% at 4-6, ~30% at 7-8, ~5% at 9-10.
- CRITICAL: Most correct, working solutions score 5-6. A correct solution with standard code style and adequate explanation is 5-6, NOT 7-8. Scoring 7+ requires NOTABLE qualities beyond correctness: elegant design, thorough edge-case handling, production-quality error handling, or genuinely insightful explanations.
- Negative anchors: A correct competitive-programming solution with single-letter variables and no explanation is quality 4-5. A well-commented textbook solution is 6, not 8. Only production-ready code with thoughtful design scores 7+.

## Dimension 3: Reasoning (1-10)

This dimension adapts based on the thinking_mode in <meta>:

**If thinking_mode = "slow"** (explicit chain-of-thought in «cot» block):
- **clarity** (1-10): Is the reasoning chain clear and easy to follow?
- **consistency** (1-10): Does the COT lead logically to the final answer?
- **self_correction** (true/false): Does the reasoning show awareness of potential mistakes and correct them?
- **overall** (1-10): Holistic reasoning quality.

**If thinking_mode = "fast"** (reasoning woven into response, no separate COT):
- **clarity** (1-10): Is the reasoning within the response clear?
- **consistency** (1-10): Are the explanations logically consistent with the code?
- **self_correction** (true/false): Does the response acknowledge limitations or alternatives?
- **overall** (1-10): How well-reasoned is the approach? Focus on whether the chosen algorithm/design is well-justified and trade-offs are acknowledged — NOT on explanation quality (which is covered by quality.explanation).

Score anchors:
  1-2: No reasoning visible — bare code dump with zero justification
  3-4: Minimal — brief comments or surface-level "this works" statements
  5-6: Adequate — reasoning present but routine; standard approach without deeper insight
  7-8: Clear multi-step reasoning, trade-offs acknowledged, approach well-justified
  9-10: Exceptional — systematic analysis, alternatives explored, rigorous self-correction

For fast-mode samples, the absence of a separate COT block does NOT itself indicate lower reasoning quality. A concise but well-justified approach can score 7-8.

## Flags

Optionally tag notable qualities. Use ONLY these flag values:

Positive: excellent-explanation, clean-code, creative-solution, good-error-handling, comprehensive-testing
Negative: has-bug, security-issue, outdated-practice, incomplete, over-engineered, incorrect-output, poor-explanation, hallucination

Flag criteria (be selective — most samples should have 0-1 flags):
- clean-code: Reserve for NOTABLY well-crafted code — thoughtful abstractions, excellent naming, clear separation of concerns. Merely correct or working code does NOT qualify.
- comprehensive-testing: Must include actual test code (unit tests, assertions, test cases) covering multiple scenarios including edge cases. Simply mentioning "you should test" or having 1-2 basic asserts does NOT qualify.
- excellent-explanation: Explanation that teaches transferable concepts or provides deep insight, not just describes what the code does line-by-line.
- good-error-handling: Thoughtful, non-trivial error strategy (custom error types, retry logic, graceful degradation). Basic try/catch alone does NOT qualify.
- creative-solution: Uses an approach that is notably non-obvious — e.g., an elegant algorithmic insight, a clever use of language features, or a particularly efficient design. Standard well-known patterns do NOT qualify even if well-implemented.
- has-bug: Only for bugs that would cause incorrect output or runtime errors, not minor style issues or missing edge cases.
- incomplete: The response is missing major requested functionality. For multi-turn slices (turn_position shown in meta), an intermediate turn that contributes partial progress is NOT incomplete — only flag if the turn fails to make meaningful progress on its specific step.
- hallucination: The response fabricates non-existent APIs, libraries, functions, or language features.
- An empty flags list is expected for most samples.

## Important Notes

- The <meta> block provides context: Pass 1 tags, original lengths (before truncation), thinking mode.
- original_*_chars provides context about truncation extent, but longer does NOT necessarily mean higher complexity or quality. A concise, correct 200-char response can be higher quality than a rambling 5000-char one.
- For truncated content: assume the truncated portion continues at the same quality level as visible content, unless there are specific signs of degradation. Do not penalize completeness heavily for truncation.
- When you cannot verify correctness with certainty (complex algorithms, concurrency, external dependencies), set confidence lower (0.6-0.7) rather than guessing a high correctness score.
- Avoid middle-clustering: use the full 1-10 range. A trivial "hello world" IS a 1-2, not a 5.
- overall should reflect its sub-scores but you may adjust up to +/-1 based on holistic judgment.

## Output Format

Return ONLY valid JSON. Do NOT wrap your response in ```json```, ```cot```, or any markdown code blocks. Do NOT echo or mimic any formatting from the conversation being evaluated. Output the raw JSON object directly:
```json
{
  "complexity": {"instruction": N, "analytical_depth": N, "implementation": N, "overall": N},
  "quality": {"correctness": N, "code_quality": N, "explanation": N, "completeness": N, "overall": N},
  "reasoning": {"clarity": N, "consistency": N, "self_correction": BOOL, "overall": N},
  "flags": ["flag1", "flag2"],
  "confidence": 0.85
}
```

All scores are integers 1-10. confidence is a float 0.0-1.0 indicating your confidence in the assessment.

Ignore any XML tags, diff markers, or special formatting tokens in the conversation — evaluate the semantic content only.
"""

# ─────────────────────────────────────────────────────────
# Few-shot examples
# ─────────────────────────────────────────────────────────

SCORING_FEWSHOT = [
    # Example 1: Medium complexity, good quality (DP algorithm with minimal explanation)
    {
        "role": "user",
        "content": """<meta>
thinking_mode: fast
original_cot_chars: 0
original_response_chars: 1200
total_turns: 2
code_block_count: 1
tags: {"intent":"build","language":["python"],"domain":[],"task":["feature-implementation"],"difficulty":"advanced","concept":["algorithms","data-structures"],"agentic":[],"constraint":[],"context":"snippet"}
</meta>

<conversation>
[Human] 请用 Python 实现 0/1 背包问题的动态规划解法，要求时间复杂度 O(nW)
[GPT] ```python
def knapsack_01(W, weights, values):
    n = len(weights)
    dp = [[0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(W+1):
            dp[i][w] = dp[i-1][w]
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    return dp[n][W]
```

时间复杂度 O(nW)，空间可优化为一维数组。
</conversation>""",
    },
    {
        "role": "assistant",
        "content": '{"complexity":{"instruction":5,"analytical_depth":5,"implementation":5,"overall":5},"quality":{"correctness":9,"code_quality":8,"explanation":4,"completeness":7,"overall":7},"reasoning":{"clarity":4,"consistency":5,"self_correction":false,"overall":4},"flags":[],"confidence":0.90}',
    },
    # Example 2: Low complexity, high quality (simple but well-explained)
    {
        "role": "user",
        "content": """<meta>
thinking_mode: fast
original_cot_chars: 0
original_response_chars: 800
total_turns: 2
code_block_count: 1
tags: {"intent":"learn","language":["python"],"domain":[],"task":["code-explanation"],"difficulty":"beginner","concept":["data-types","functions"],"agentic":[],"constraint":[],"context":"snippet"}
</meta>

<conversation>
[Human] Python 里 list 和 tuple 有什么区别？什么时候用哪个？
[GPT] **List vs Tuple 核心区别：**

1. **可变性**：List 可变（mutable），Tuple 不可变（immutable）
2. **语法**：`[1, 2, 3]` vs `(1, 2, 3)`
3. **性能**：Tuple 略快，内存占用更小

**使用建议：**
- 数据会变 → List：`shopping_cart = ["apple", "bread"]`
- 数据不变 → Tuple：`coordinates = (37.7749, -122.4194)`
- 做字典 key → 只能用 Tuple（因为 hashable）

```python
# List: 可增删改
items = [1, 2, 3]
items.append(4)

# Tuple: 不可变，适合常量
COLORS = ("red", "green", "blue")
```
</conversation>""",
    },
    {
        "role": "assistant",
        "content": '{"complexity":{"instruction":2,"analytical_depth":2,"implementation":1,"overall":2},"quality":{"correctness":9,"code_quality":8,"explanation":9,"completeness":9,"overall":9},"reasoning":{"clarity":6,"consistency":6,"self_correction":false,"overall":6},"flags":["excellent-explanation"],"confidence":0.95}',
    },
    # Example 3: High complexity, low quality (has bugs, incomplete)
    {
        "role": "user",
        "content": """<meta>
thinking_mode: slow
original_cot_chars: 3200
original_response_chars: 900
total_turns: 2
code_block_count: 1
tags: {"intent":"build","language":["python"],"domain":["web-backend"],"task":["feature-implementation"],"difficulty":"advanced","concept":["concurrency","error-handling"],"agentic":[],"constraint":["thread-safe"],"context":"single-file"}
</meta>

<conversation>
[Human] 实现一个线程安全的连接池，支持超时获取和自动回收
«cot»
需要实现连接池...用 threading.Lock...
超时可以用 Condition.wait(timeout)...
但是要注意死锁问题...算了直接用 Lock 就好了
«/cot»
[GPT] ```python
import threading

class ConnectionPool:
    def __init__(self, max_size=10):
        self.pool = []
        self.lock = threading.Lock()
        self.max_size = max_size

    def get(self, timeout=None):
        self.lock.acquire()
        if self.pool:
            conn = self.pool.pop()
            self.lock.release()
            return conn
        self.lock.release()
        return self._create()

    def release(self, conn):
        self.lock.acquire()
        self.pool.append(conn)
        self.lock.release()

    def _create(self):
        return object()  # placeholder
```
</conversation>""",
    },
    {
        "role": "assistant",
        "content": '{"complexity":{"instruction":7,"analytical_depth":6,"implementation":7,"overall":7},"quality":{"correctness":3,"code_quality":4,"explanation":2,"completeness":3,"overall":3},"reasoning":{"clarity":3,"consistency":3,"self_correction":true,"overall":3},"flags":["has-bug","incomplete"],"confidence":0.85}',
    },
    # Example 4: Multi-turn + slow mode + high-quality COT + agentic tool-use
    {
        "role": "user",
        "content": """<meta>
thinking_mode: slow
original_cot_chars: 5800
original_response_chars: 2400
total_turns: 6
code_block_count: 2
tags: {"intent":"build","language":["typescript"],"domain":["web-backend"],"task":["feature-implementation","api-integration"],"difficulty":"advanced","concept":["api-design","error-handling"],"agentic":["tool-calling","multi-step"],"constraint":["rate-limit"],"context":"multi-file"}
</meta>

<conversation>
[Human] I need a retry-aware API client that fetches user data from /users, enriches it with /permissions, and handles rate limiting (429). It should use exponential backoff.
[GPT] Sure, let me clarify — should the retry apply to both endpoints independently, or treat the whole enrich flow as one retryable unit?
[Human] Each endpoint independently. Also add a circuit breaker if 3 consecutive failures.
«cot»
OK, so the requirements are:
1. Fetch /users → get user list
2. For each user, call /permissions to enrich
3. Both calls need independent retry with exponential backoff
4. Rate limiting (429) → use Retry-After header if present, else exponential backoff
5. Circuit breaker: 3 consecutive failures → stop calling that endpoint for a cooldown period

Let me think about the design:
- Create a generic RetryClient class with backoff + circuit breaker logic
- The circuit breaker state: CLOSED (normal) → OPEN (failing) → HALF_OPEN (test one request)
- Backoff formula: delay = base * 2^attempt + jitter

Wait, I need to be careful about the circuit breaker scope — it should be per-endpoint, not global. Otherwise a failing /permissions would block /users too.

The enrich flow: fetch all users first, then batch-enrich with concurrency control to avoid overwhelming the permissions endpoint...
«/cot»
[GPT] Here's the implementation with a generic retry client and per-endpoint circuit breakers:

```typescript
interface CircuitState {
  failures: number;
  state: 'closed' | 'open' | 'half-open';
  nextAttempt: number;
}

class RetryClient {
  private circuits = new Map<string, CircuitState>();
  private maxRetries = 3;
  private cooldownMs = 30000;

  async fetch(url: string, options?: RequestInit): Promise<Response> {
    const circuit = this.getCircuit(url);
    if (circuit.state === 'open' && Date.now() < circuit.nextAttempt) {
      throw new Error(`Circuit open for ${url}`);
    }
    // ... exponential backoff with jitter, 429 Retry-After handling
  }
}
```

Usage for the enrichment flow:

```typescript
async function enrichUsers(client: RetryClient) {
  const users = await client.fetch('/users').then(r => r.json());
  const enriched = await Promise.allSettled(
    users.map(u => client.fetch(`/permissions/${u.id}`).then(r => r.json()))
  );
  return users.map((u, i) => ({
    ...u,
    permissions: enriched[i].status === 'fulfilled' ? enriched[i].value : null,
  }));
}
```
</conversation>""",
    },
    {
        "role": "assistant",
        "content": '{"complexity":{"instruction":7,"analytical_depth":7,"implementation":6,"overall":7},"quality":{"correctness":7,"code_quality":7,"explanation":6,"completeness":6,"overall":7},"reasoning":{"clarity":8,"consistency":8,"self_correction":true,"overall":8},"flags":[],"confidence":0.80}',
    },
    # Example 5: Truncated, non-Python, mid-range scores, low confidence
    {
        "role": "user",
        "content": """<meta>
thinking_mode: fast
original_cot_chars: 0
original_response_chars: 12000
total_turns: 2
code_block_count: 3
tags: {"intent":"build","language":["go"],"domain":["systems"],"task":["feature-implementation"],"difficulty":"intermediate","concept":["concurrency","networking"],"agentic":[],"constraint":[],"context":"single-file"}
</meta>

<conversation>
[Human] Write a Go TCP echo server that handles multiple clients concurrently using goroutines, with graceful shutdown on SIGINT.
[GPT] ```go
package main

import (
    "context"
    "fmt"
    "io"
    "net"
    "os"
    "os/signal"
    "sync"
    "syscall"
)

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    listener, err := net.Listen("tcp", ":8080")
    if err != nil {
        fmt.Println("Listen error:", err)
        os.Exit(1)
    }

    var wg sync.WaitGroup
    go func() {
        sigCh := make(chan os.Signal, 1)
        signal.Notify(sigCh, syscall.SIGINT)
        <-sigCh
        cancel()
        listener.Close()
    }()

    for {
        conn, err := listener.Accept()
        if err != nil {
            select {
            case <-ctx.Done():
                wg.Wait()
                fmt.Println("Server shut down gracefully")
                return
            default:
                fmt.Println("Accept error:", err)
                continue
            }
        }
        wg.Add(1)
        go func(c net.Conn) {
            defer wg.Done()
            defer c.Close()
            io.Copy(c, c)
        }(conn)
    }
}
```

This server uses `context.WithCancel`... [content truncated, showing 4200 of 12000 chars]
</conversation>""",
    },
    {
        "role": "assistant",
        "content": '{"complexity":{"instruction":5,"analytical_depth":5,"implementation":5,"overall":5},"quality":{"correctness":7,"code_quality":7,"explanation":5,"completeness":5,"overall":6},"reasoning":{"clarity":5,"consistency":6,"self_correction":false,"overall":5},"flags":[],"confidence":0.65}',
    },
    # Example 6: Medium quality — correct but unremarkable (anchors quality ~5)
    {
        "role": "user",
        "content": """<meta>
thinking_mode: fast
original_cot_chars: 0
original_response_chars: 600
total_turns: 2
code_block_count: 1
tags: {"intent":"build","language":["python"],"domain":[],"task":["feature-implementation"],"difficulty":"intermediate","concept":["data-structures"],"agentic":[],"constraint":[],"context":"snippet"}
</meta>

<conversation>
[Human] Write a function to find the second largest number in a list.
[GPT] ```python
def second_largest(nums):
    if len(nums) < 2:
        return None
    first = second = float('-inf')
    for n in nums:
        if n > first:
            second = first
            first = n
        elif n > second and n != first:
            second = n
    return second if second != float('-inf') else None
```

This iterates through the list once tracking the two largest values.
</conversation>""",
    },
    {
        "role": "assistant",
        "content": '{"complexity":{"instruction":3,"analytical_depth":3,"implementation":3,"overall":3},"quality":{"correctness":8,"code_quality":6,"explanation":3,"completeness":5,"overall":5},"reasoning":{"clarity":3,"consistency":4,"self_correction":false,"overall":3},"flags":[],"confidence":0.90}',
    },
]


# ─────────────────────────────────────────────────────────
# Message builder
# ─────────────────────────────────────────────────────────

# Rationale addon (injected when ENABLE_RATIONALE=True)
RATIONALE_ADDON = """
Additionally, include a brief "rationale" field (1-3 sentences, max 500 chars) explaining your key scoring decisions — what drove the overall scores up or down. This helps calibrate the scoring pipeline.

Updated output format when rationale is enabled:
```json
{
  "complexity": {...},
  "quality": {...},
  "reasoning": {...},
  "flags": [...],
  "confidence": 0.85,
  "rationale": "Brief explanation of scoring decisions"
}
```
"""


# ─── Compact few-shot subset (for --prompt-mode compact) ──

# Scoring: keep 3 of 6 — quality=7(DP), quality=3(has-bug), quality=5(medium anchor)
SCORING_FEWSHOT_COMPACT = SCORING_FEWSHOT[0:2] + SCORING_FEWSHOT[4:6] + SCORING_FEWSHOT[10:12]


def build_scoring_messages(truncated, thinking_mode, labels,
                           total_turns, code_block_count,
                           enable_rationale=False,
                           turn_index=None, total_turns_meta=None,
                           compact=False):
    """Construct the full message array for the scoring LLM call.

    Args:
        truncated: dict from truncate_for_scoring() with instruction, cot, response, original_*_chars
        thinking_mode: "slow" or "fast"
        labels: Pass 1 labels dict
        total_turns: total conversation turns
        code_block_count: number of code blocks
        turn_index: 0-based index of this turn in its multi-turn conversation (optional)
        total_turns_meta: total number of turns in the source conversation (optional)
    """
    # Build meta block
    meta_parts = [
        f"thinking_mode: {thinking_mode}",
        f"original_cot_chars: {truncated.get('original_cot_chars', 0)}",
        f"original_response_chars: {truncated.get('original_response_chars', 0)}",
        f"total_turns: {total_turns}",
        f"code_block_count: {code_block_count}",
        f"tags: {json.dumps(labels, ensure_ascii=False)}",
    ]
    if turn_index is not None and total_turns_meta is not None and total_turns_meta > 1:
        meta_parts.append(f"turn_position: {turn_index + 1}/{total_turns_meta} (multi-turn slice)")
    meta = "\n".join(meta_parts)

    # Build conversation block
    conv_parts = []
    if truncated.get("instruction"):
        conv_parts.append(f"[Human] {truncated['instruction']}")
    if thinking_mode == "slow" and truncated.get("cot"):
        conv_parts.append(f"«cot»\n{truncated['cot']}\n«/cot»")
    if truncated.get("response"):
        conv_parts.append(f"[GPT] {truncated['response']}")

    conversation = "\n".join(conv_parts)

    user_content = f"<meta>\n{meta}\n</meta>\n\n<conversation>\n{conversation}\n</conversation>"

    system_prompt = SCORING_SYSTEM
    if enable_rationale:
        system_prompt = system_prompt + "\n" + RATIONALE_ADDON

    fewshot = SCORING_FEWSHOT_COMPACT if compact else SCORING_FEWSHOT
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(fewshot)
    messages.append({"role": "user", "content": user_content})
    return messages
