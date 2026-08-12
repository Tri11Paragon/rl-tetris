import tetris
import MaTris.matris as matris
import numpy as np
import time
from config import Config
from pathlib import Path

ACTION_SIZE = 5
TEST_COUNT = 10000

def main():
    action_data = np.concat([np.arange(ACTION_SIZE)] * TEST_COUNT)
    print(action_data.shape)
    np.random.shuffle(action_data)

    config = Config(Path(__file__).parent / "nix" / "defaults.nix")
    loaded_config = config.load()

    tetris_engine = tetris.PyTetrisEngine(time.time_ns(), loaded_config.json_str)
    matris_engine = matris.Matris(loaded_config)

    tetris_start = time.time_ns()
    for action in action_data:
        state, reward, lines_cleared, game_over, truncated = tetris_engine.step(action)

        if game_over:
            tetris_engine.reset()
    tetris_end = time.time_ns()

    matris_start = time.time_ns()
    for action in action_data:
        state, reward, lines_cleared, game_over, truncated = matris_engine.step(action)

        if game_over:
            matris_engine.reset()
    matris_end = time.time_ns()

    tetris_difference = tetris_end - tetris_start
    matris_difference = matris_end - matris_start

    print(f"Tetris (new): {tetris_difference} ns || {tetris_difference / 1e6} ms")
    print(f"MaTris (old): {matris_difference} ns || {matris_difference / 1e6} ms")

if __name__ == "__main__":
    main()