"""Fixture manifest checks to catch accounting drift."""

from pathlib import Path


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "e2e_folder_test"

EXPECTED_FILES = {
    "code/commitpackft_multilang.jsonl": 200,
    "code/magicoder_oss_instruct.jsonl": 200,
    "code/mot_code_part1.jsonl": 200,
    "code/mot_code_part2.jsonl": 200,
    "multi_turn/code_feedback_multiturn.jsonl": 75,
    "multi_turn/coderforge_swe_trajectories.jsonl": 8,
    "multi_turn/nemotron_agentless_file_lookup.jsonl": 100,
    "multi_turn/nemotron_swe_repair.jsonl": 100,
}


def test_e2e_fixture_manifest_counts():
    actual_files = {
        str(path.relative_to(FIXTURE_ROOT))
        for path in FIXTURE_ROOT.rglob("*.jsonl")
    }
    assert actual_files == set(EXPECTED_FILES)

    for rel_path, expected_lines in EXPECTED_FILES.items():
        path = FIXTURE_ROOT / rel_path
        with open(path, "r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        assert line_count == expected_lines
