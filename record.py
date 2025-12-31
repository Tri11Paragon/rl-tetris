import MaTris.matris as tetris
import pygame

import math
import pathlib
import random

import pickle

from MaTris.matris import GameOver
from MaTris.matris import MATRIX_WIDTH

from experience import Experience

def main():
    print("Starting game")
    pygame.init()

    screen = pygame.display.set_mode((tetris.WIDTH, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")
    matris = tetris.Matris()
    game = tetris.Game()
    game.main(screen, matris)

    buffer = []

    state = matris.reset()
    try:
        while True:
            game.clock.tick(50)
            # best_actions = matris.best_action_set()
            # # print(best_actions)
            # for action in best_actions:
            #     next_state, reward, game_over = matris.step(action, decay=False)
            #     game.redraw()
            #     buffer.append(Experience(state, action.value, reward, next_state, 0 if game_over else 1))
            #     state = next_state
            #     if game_over:
            #         print("We somehow got a game over while picking the best option. What happened?")
            #         state = matris.reset()
            actions = game.get_user_actions()

            # print(matris.get_columns_state())
            # print()
            # column = random.randint(0, MATRIX_WIDTH - 1)
            # next_state, reward, game_over = matris.place_in_column(column)
            # if game_over:
            #     state = matris.reset()

            if len(actions) == 0:
                game.redraw()
                continue
            for action in actions:
                next_state, reward, game_over = matris.step(action)
                game.redraw()
                buffer.append(Experience(state, action.value, reward, next_state, 0 if game_over else 1))
                state = next_state
                if game_over:
                    state = matris.reset()
    except Exception as e:
        print(e)
        raise e

    print(f"Saving action state responses {len(buffer)}")
    with open("actions.pkl", "wb") as f:
        pickle.dump(buffer, f)

main()