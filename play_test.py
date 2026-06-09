import pygame

import MaTris.matris as tetris
from experience import Experience

def main():
    pygame.init()
    screen = pygame.display.set_mode((tetris.WIDTH, tetris.HEIGHT))
    pygame.display.set_caption("MaTris")
    matris = tetris.Matris()
    game = tetris.Game()
    game.main(screen, matris)
    
    state = matris.reset()
    rewards = []
    lines = []
    scores = []
    total_reward = 0
    try:
        while True:
            game.clock.tick(50)
            game.get_user_actions()
            best_actions = matris.best_action_set()
            if len(best_actions) == 0:
                game.redraw()
                continue
            # print(best_actions)
            for action in best_actions:
                next_state, reward, game_over = matris.step(action)
                game.redraw()
                # buffer.append(Experience(state, action.value, reward, next_state, 0 if game_over else 1))
                # if game_over:
                    # print(state)
                total_reward += reward
                state = next_state
                if game_over:
                    rewards.append(total_reward)
                    lines.append(matris.lines)
                    scores.append(matris.score)
                    print(f"Game Over {total_reward} || Lines Cleared: {matris.lines} || Score: {matris.score}")
                    state = matris.reset()
                    total_reward = 0
    except:
        pass
    print(
        f"Average Reward {sum(rewards) / len(rewards)} || Average Lines Cleared: {sum(lines) / len(lines)} || Num of episodes: {len(lines)}")
    
    
main()