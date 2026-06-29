import json
import os
import subprocess
from pathlib import Path


class DotDict:
    def __init__(self, mapping):
        for k, v in mapping.items():
            if isinstance(v, dict):
                v = DotDict(v)
            setattr(self, k, v)


class Config:
    def __init__(self, config_file):
        nix_path = os.getenv("NIX_CONFIG_PATH", None)
        if nix_path is None:
            raise ValueError("NIX_CONFIG_PATH environment variable not set. Please open a nix shell.")

        self.file = Path(config_file)

        completed = subprocess.run(
            ["nix-instantiate", "--eval", nix_path, "--argstr", "file", f"{self.file.absolute()}", "--show-trace", "--json", "--strict"],
            capture_output=True,
            text=True,
            check=False
        )

        if completed.returncode != 0:
            print(completed.stderr)
            raise RuntimeError("Failed to instantiate Nix expression")

        self.config = json.loads(completed.stdout)

    def load(self) -> DotDict:
        return DotDict(self.config)