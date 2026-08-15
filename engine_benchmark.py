import tetris
from tqdm.auto import tqdm

import MaTris.matris as matris
import numpy as np
import time
from config import Config
from pathlib import Path

ACTION_SIZE = 5
TEST_COUNT = 1000

def validate_tuple(action, tetris_tuple, matris_tuple, should_exit: bool = False) -> bool:
    tuple_names = ["state", "reward", "lines_cleared", "game_over", "truncated"]
    matris_state = matris_tuple[0]
    matris_state = (matris_state != 0).astype(np.float32)
    tuple_equals = [(tetris_tuple[0] == matris_state).all()]
    tuple_equals = tuple_equals + [mat == tet for mat, tet in zip(matris_tuple[1:], tetris_tuple[1:])]
    tuple_data = {name: value for name, value in zip(tuple_names, tuple_equals)}

    if not all(tuple_equals):
        print()
        print("Tetris Tuple")
        print(tetris_tuple)
        print()
        print("MaTris Tuple")
        print(matris_state, matris_tuple[1:])
        print()
        print(matris.Action(action))
        print(f"Tuple not equal: {tuple_data}")
        if should_exit:
            exit(0)
        return False
    return True

def eval_time_to_clear(action_data, tetris_engine, matris_engine):
    tetris_start = time.time_ns()
    for action in action_data:
        state, reward, lines_cleared, game_over, truncated = tetris_engine.step(action)

        if game_over:
            tetris_engine.reset()
    tetris_end = time.time_ns()

    matris_start = time.time_ns()
    for action in action_data:
        state, reward, lines_cleared, game_over, truncated = matris_engine.step(int(action))

        if game_over:
            matris_engine.reset()
    matris_end = time.time_ns()

    tetris_difference = tetris_end - tetris_start
    matris_difference = matris_end - matris_start

    print(f"Tetris (new): {tetris_difference} ns || {tetris_difference / 1e6} ms")
    print(f"MaTris (old): {matris_difference} ns || {matris_difference / 1e6} ms")

def main():
    action_data = np.concat([np.arange(ACTION_SIZE)] * TEST_COUNT)
    print(action_data.shape)
    np.random.shuffle(action_data)

    config = Config(Path(__file__).parent / "nix" / "defaults.nix")
    loaded_config = config.load()

    tetris_engine = tetris.PyTetrisEngine(time.time_ns(), loaded_config.json_str)
    matris_engine = matris.Matris(loaded_config)

    matris_state = matris_engine.current_state()
    matris_state = (matris_state != 0).astype(np.float32)
    tuple_equals = (tetris_engine.current_state() == matris_state).all()
    if not tuple_equals:
        print("Matris and Tetris beginning states are not equal")

    # eval_time_to_clear(action_data, tetris_engine, matris_engine)

    for action in tqdm(action_data):
        print()
        print(matris.Action(action))
        tetris_tuple = tetris_engine.step(action)
        matris_tuple = matris_engine.step(int(action))

        validate_tuple(action, tetris_tuple, matris_tuple, should_exit=True)

if __name__ == "__main__":
    main()