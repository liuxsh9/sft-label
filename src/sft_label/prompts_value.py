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
- **reasoning** (1-10): How many logical steps or how deep the reasoning needed?
- **implementation** (1-10): How difficult is the actual code implementation?
- **overall** (1-10): Holistic complexity assessment.

Score anchors:
  1-2: Trivial — print("hello"), variable assignment, basic syntax questions
  3-4: Basic — simple CRUD operations, basic data manipulation, single-function tasks
  5-6: Intermediate — design pattern application, moderate algorithms (sorting, BFS/DFS), multi-step logic with error handling
  7-8: Advanced — system design, concurrent programming, complex algorithms (DP, graph theory), multi-module architecture
  9-10: Expert — novel algorithm design, compiler/interpreter implementation, kernel-level programming, cutting-edge technique application

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

## Dimension 3: Reasoning (1-10)

This dimension adapts based on the thinking_mode in <meta>:

**If thinking_mode = "slow"** (explicit chain-of-thought in [COT] block):
- **clarity** (1-10): Is the reasoning chain clear and easy to follow?
- **consistency** (1-10): Does the COT lead logically to the final answer?
- **self_correction** (true/false): Does the reasoning show awareness of potential mistakes and correct them?
- **overall** (1-10): Holistic reasoning quality.

**If thinking_mode = "fast"** (reasoning woven into response, no separate COT):
- **clarity** (1-10): Is the reasoning within the response clear?
- **consistency** (1-10): Are the explanations logically consistent with the code?
- **self_correction** (true/false): Does the response acknowledge limitations or alternatives?
- **overall** (1-10): How well-integrated is the reasoning with the implementation?

## Flags

Optionally tag notable qualities. Use ONLY these flag values:

Positive: excellent-explanation, clean-code, creative-solution, good-error-handling, comprehensive-testing
Negative: has-bug, security-issue, outdated-practice, incomplete, over-engineered, incorrect-output, poor-explanation

Only include flags that clearly apply. An empty flags list is fine.

## Important Notes

- The <meta> block provides context: Pass 1 tags, original lengths (before truncation), thinking mode.
- If content was truncated, consider original_*_chars for complexity assessment — longer COT/response often indicates higher complexity.
- For correctness: if you cannot fully assess correctness due to truncation, lean toward the score suggested by the visible code quality and approach correctness. Do not penalize heavily for truncation.
- Avoid middle-clustering: use the full 1-10 range. A trivial "hello world" IS a 1-2, not a 5.

## Output Format

Return ONLY valid JSON:
```json
{
  "complexity": {"instruction": N, "reasoning": N, "implementation": N, "overall": N},
  "quality": {"correctness": N, "code_quality": N, "explanation": N, "completeness": N, "overall": N},
  "reasoning": {"clarity": N, "consistency": N, "self_correction": BOOL, "overall": N},
  "flags": ["flag1", "flag2"],
  "confidence": 0.85
}
```

All scores are integers 1-10. confidence is a float 0.0-1.0 indicating your confidence in the assessment.
"""

# ─────────────────────────────────────────────────────────
# Few-shot examples
# ─────────────────────────────────────────────────────────

SCORING_FEWSHOT = [
    # Example 1: High complexity, high quality (DP algorithm with good explanation)
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
        "content": '{"complexity":{"instruction":6,"reasoning":5,"implementation":5,"overall":6},"quality":{"correctness":9,"code_quality":8,"explanation":5,"completeness":7,"overall":7},"reasoning":{"clarity":6,"consistency":8,"self_correction":false,"overall":7},"flags":["clean-code"],"confidence":0.90}',
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
        "content": '{"complexity":{"instruction":2,"reasoning":2,"implementation":1,"overall":2},"quality":{"correctness":9,"code_quality":8,"explanation":9,"completeness":9,"overall":9},"reasoning":{"clarity":9,"consistency":9,"self_correction":false,"overall":9},"flags":["excellent-explanation"],"confidence":0.95}',
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
[COT]
需要实现连接池...用 threading.Lock...
超时可以用 Condition.wait(timeout)...
但是要注意死锁问题...算了直接用 Lock 就好了
[/COT]
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
        "content": '{"complexity":{"instruction":8,"reasoning":6,"implementation":7,"overall":7},"quality":{"correctness":3,"code_quality":4,"explanation":2,"completeness":3,"overall":3},"reasoning":{"clarity":3,"consistency":3,"self_correction":false,"overall":3},"flags":["has-bug","incomplete"],"confidence":0.85}',
    },
]


# ─────────────────────────────────────────────────────────
# Message builder
# ─────────────────────────────────────────────────────────

def build_scoring_messages(truncated, thinking_mode, labels,
                           total_turns, code_block_count):
    """Construct the full message array for the scoring LLM call.

    Args:
        truncated: dict from truncate_for_scoring() with instruction, cot, response, original_*_chars
        thinking_mode: "slow" or "fast"
        labels: Pass 1 labels dict
        total_turns: total conversation turns
        code_block_count: number of code blocks
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
    meta = "\n".join(meta_parts)

    # Build conversation block
    conv_parts = []
    if truncated.get("instruction"):
        conv_parts.append(f"[Human] {truncated['instruction']}")
    if thinking_mode == "slow" and truncated.get("cot"):
        conv_parts.append(f"[COT]\n{truncated['cot']}\n[/COT]")
    if truncated.get("response"):
        conv_parts.append(f"[GPT] {truncated['response']}")

    conversation = "\n".join(conv_parts)

    user_content = f"<meta>\n{meta}\n</meta>\n\n<conversation>\n{conversation}\n</conversation>"

    messages = [{"role": "system", "content": SCORING_SYSTEM}]
    messages.extend(SCORING_FEWSHOT)
    messages.append({"role": "user", "content": user_content})
    return messages
