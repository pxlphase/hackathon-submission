from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS = REPO_ROOT / "artifacts/inputs"
OUTPUTS = REPO_ROOT / "artifacts/outputs"

TEST_CASES = sorted(p.stem for p in INPUTS.glob("test_case_*.txt"))


def test_inputs_found():
    assert TEST_CASES, f"no test_case_<n>.txt inputs found in {INPUTS}"


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_output_exists(test_case):
    output = OUTPUTS / f"{test_case}.stp"
    assert output.is_file(), f"missing output {output.relative_to(REPO_ROOT)}"


def test_no_unexpected_outputs():
    expected = {f"{test_case}.stp" for test_case in TEST_CASES} | {".gitkeep"}
    unexpected = {p.name for p in OUTPUTS.iterdir()} - expected
    assert not unexpected, f"unexpected files in outputs (names must be test_case_<n>.stp): {sorted(unexpected)}"
