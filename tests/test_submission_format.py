from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS = REPO_ROOT / "artifacts/inputs"
OUTPUTS = REPO_ROOT / "artifacts/outputs"

N_VIEWS = 8

TEST_CASES = sorted(
    (p.name for p in INPUTS.glob("test_case_*") if (p / "description.md").is_file()),
    key=lambda name: int(name.removeprefix("test_case_")),
)
# A fresh repo from the template has no outputs yet; check them once any are committed
HAS_OUTPUTS = any(OUTPUTS.glob("test_case_*.stp"))


def test_inputs_found():
    assert TEST_CASES, f"no test_case_<n>/description.md inputs found in {INPUTS}"


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_inputs_complete(test_case):
    views = sorted((INPUTS / test_case / "images").glob("view_*.png"))
    assert len(views) == N_VIEWS, f"{test_case}: expected {N_VIEWS} views in images/, found {len(views)}"


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_output_exists(test_case):
    output = OUTPUTS / f"{test_case}.stp"
    assert output.is_file(), f"missing output {output.relative_to(REPO_ROOT)}"


def test_no_unexpected_outputs():
    expected = {f"{test_case}.stp" for test_case in TEST_CASES} | {".gitkeep"}
    unexpected = {p.name for p in OUTPUTS.iterdir()} - expected
    assert not unexpected, f"unexpected files in outputs (names must be test_case_<n>.stp): {sorted(unexpected)}"
