import argparse
from dataclasses import dataclass
from pathlib import Path

from agentic_cad.example import MockAgent

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = REPO_ROOT / "artifacts/inputs"
DEFAULT_OUTPUTS = REPO_ROOT / "artifacts/outputs"


@dataclass
class TestCase:
    name: str  # test_case_<n>
    description: str  # contents of description.md
    views: list[Path]  # images/view_0_front.png ... view_7_iso_back_left_bottom.png
    context: Path | None  # context.png, only present for parts taken from an assembly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUTS)

    return parser.parse_args()


def load_test_cases(input_dir: Path) -> list[TestCase]:
    test_cases = []
    for case_dir in input_dir.glob("test_case_*"):
        if not (case_dir / "description.md").is_file():
            continue
        context = case_dir / "context.png"
        test_cases.append(TestCase(
            name=case_dir.name,
            description=(case_dir / "description.md").read_text(),
            views=sorted((case_dir / "images").glob("view_*.png")),
            context=context if context.is_file() else None,
        ))
    return sorted(test_cases, key=lambda tc: int(tc.name.removeprefix("test_case_")))


def main(args):
    print("Initialising agentic workflow")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent = MockAgent()
    for test_case in load_test_cases(args.input_dir):
        print(f"processing {test_case.name}")
        output_path = args.output_dir / f"{test_case.name}.stp"
        agent.generate_cad(
            prompt=test_case.description,
            views=test_case.views,
            context=test_case.context,
            output_path=output_path,
        )

    print("Pipeline generated all CAD designs")


if __name__ == "__main__":
    args = parse_args()
    main(args)
