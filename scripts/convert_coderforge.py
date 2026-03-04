"""One-off: Convert CoderForge-Preview SWE_Rebench trajectories to Pangu format.
Handles OpenHands-style tool_calls → <tool_call> serialization and think → COT.
"""
import json, random

N = 100
SEED = 42
OUTPUT = "tests/fixtures/e2e_folder_test/multi_turn/coderforge_swe_trajectories.json"

def convert_trajectory(traj, idx):
    """Convert one CoderForge trajectory to Pangu format."""
    msgs_raw = traj["messages"]
    msgs = json.loads(msgs_raw) if isinstance(msgs_raw, str) else msgs_raw
    tools_raw = traj.get("tools", "[]")
    tools = json.loads(tools_raw) if isinstance(tools_raw, str) else (tools_raw or [])

    pangu_data = []
    for m in msgs:
        role = m["role"]
        content = m.get("content") or ""
        tool_calls = m.get("tool_calls")

        # Map roles
        if role == "system":
            pangu_role = "system"
        elif role == "user":
            pangu_role = "user"
        elif role == "assistant":
            pangu_role = "assistant"
        elif role == "tool":
            pangu_role = "tool"
        else:
            pangu_role = role

        # Handle tool_calls in assistant messages
        if tool_calls:
            parts = []
            if content:
                parts.append(content)
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "unknown")
                fn_args = fn.get("arguments", "{}")
                # Parse arguments if string
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {"raw": fn_args}

                # Convert "think" tool to COT markers
                if fn_name == "think":
                    thought = fn_args.get("thought", fn_args.get("content", str(fn_args)))
                    parts.append(f"[unused16]{thought}[unused17]")
                else:
                    tc_obj = {"name": fn_name, "arguments": fn_args}
                    parts.append(f"<tool_call>\n{json.dumps(tc_obj, ensure_ascii=False)}\n</tool_call>")

            content = "\n".join(parts)

        pangu_data.append({"role": pangu_role, "content": content})

    result = {
        "id": f"coderforge-{idx}",
        "source": "togethercomputer/CoderForge-Preview",
        "data": pangu_data,
    }

    # Store tool definitions if present
    if tools:
        tool_defs = []
        for t in tools:
            fn = t.get("function", t)
            tool_defs.append(f"{fn.get('name', '?')}: {fn.get('description', '')[:100]}")
        result["tools"] = "\n".join(tool_defs)

    return result


print("Loading CoderForge-Preview SWE_Rebench...")
from datasets import load_dataset

ds = load_dataset(
    "togethercomputer/CoderForge-Preview",
    "trajectories",
    split="SWE_Rebench",
    streaming=True,
)

# Collect samples (streaming, so we need to iterate)
all_trajs = []
for i, sample in enumerate(ds):
    all_trajs.append(sample)
    if i % 5000 == 0 and i > 0:
        print(f"  loaded {i}...")
print(f"Total trajectories: {len(all_trajs)}")

# Sample
random.seed(SEED)
sampled = random.sample(all_trajs, min(N, len(all_trajs)))

# Convert
out = []
turn_counts = []
for i, traj in enumerate(sampled):
    converted = convert_trajectory(traj, i)
    out.append(converted)
    turn_counts.append(len(converted["data"]))

with open(OUTPUT, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# Stats
avg_turns = sum(turn_counts) / len(turn_counts)
rewards = [t.get("reward", 0) for t in sampled]
rewards_f = [float(r) if r is not None else 0 for r in rewards]
resolved = sum(1 for r in rewards_f if r > 0)
print(f"\nOutput: {len(out)} trajectories → {OUTPUT}")
print(f"Avg turns per trajectory: {avg_turns:.1f}")
print(f"Resolved (reward > 0): {resolved}/{len(out)} ({100*resolved/len(out):.0f}%)")
print(f"Turn range: {min(turn_counts)}-{max(turn_counts)}")
