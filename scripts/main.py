import argparse

from pathlib import Path
from agentic_cad.example import MockAgent

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = REPO_ROOT / "artifacts/inputs"
DEFAULT_OUTPUTS = REPO_ROOT / "artifacts/outputs"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUTS)

    return parser.parse_args()

def read_txt(file_path: Path):
    pass

def read_img(file_path: Path):
    pass

def main(args):
    print("Initialising agentic workflow")

    agent = MockAgent()
    for p in args.input_dir.glob("*.txt"):
        print(f"processing {p}")
        prompt = read_txt(p)
        img = read_img(p.with_suffix(".jpg"))
        output_path = DEFAULT_OUTPUTS / f"{p.stem}.stp"
        agent.generate_cad(prompt=prompt, img=img, output_path=output_path)

    print("Pipeline generated all CAD designs")


if __name__ == "__main__":
    args = parse_args()
    main(args)
