from pathlib import Path


class MockAgent:
    def __init__(self):
        pass

    def generate_cad(self, prompt: str, views: list[Path], context: Path | None, output_path: Path) -> None:
        print("thinking hard.....")

        mock_step = "this is mock step content"

        with open(output_path, "w") as f:
            f.write(mock_step)

        print("successful generation")
