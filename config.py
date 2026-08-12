import json
import os
import subprocess
from pathlib import Path

from typing import Any

import config_types

ItemNotFound = object()

class DotDict(config_types.NNConfig):
    def __init__(self, mapping):
        self.json_str = ""
        for k, v in mapping.items():
            if isinstance(v, dict):
                v = DotDict(v)
            setattr(self, k, v)

    def _resolve(self, location: str, parts: list[str]) -> Any:
        dotdict = self
        for part in parts:
            if not hasattr(dotdict, part):
                raise ValueError(f"Invalid Key: {location}. Failed to resolve {part}")
            dotdict = getattr(dotdict, part)
        return dotdict

    def resolve(self, location: str):
        return self._resolve(location, location.split("."))

    def find(self, item: str):
        try:
            return self.resolve(item)
        except ValueError:
            return ItemNotFound

    def __getattr__(self, item: str):
        return self.resolve(item)


class Config:
    def __init__(self, config_file):
        nix_path = os.getenv("NIX_CONFIG_PATH", None)
        if nix_path is None:
            raise ValueError("NIX_CONFIG_PATH environment variable not set. Please open a nix shell.")

        self.file = Path(config_file)
        if not self.file.exists():
            with self.file.open("w") as f:
                f.write("helpers: with helpers; {}")
                f.flush()

        completed = subprocess.run(
            ["nix-instantiate", "--eval", nix_path, "--argstr", "file", f"{self.file.absolute()}", "--show-trace", "--json", "--strict"],
            capture_output=True,
            text=True,
            check=False
        )

        if completed.returncode != 0:
            print(completed.stderr)
            exit(1)

        nix_data = json.loads(completed.stdout)

        self.config = nix_data["config_data"]

        script_dir = os.path.dirname(os.path.abspath(__file__))

        with open(script_dir + "/config_types.py", "w") as f:
            f.write("from dataclasses import dataclass\n")
            f.write(nix_data["python"])

        with open(script_dir + "/engine/src/types.rs", "w") as f:
            f.write(nix_data["rust"])

    def load(self) -> DotDict:
        dot = DotDict(self.config)
        dot.json_str = json.dumps(self.config)
        return dot