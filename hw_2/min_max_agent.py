import copy
from typing import Tuple
import numpy as np
from tic_tac_toe import TicTacToe


class MinMaxAgent(object):
    WIN_VALUE = 1
    DRAW_VALUE = 0
    LOSS_VALUE = -1

    def __init__(self, env):
        self.side = None
        self.cache = {}

    def new_game(self, side: int):
        if self.side != side:
            self.side = side
            self.cache = {}

    def get_next_step(self, env,  train):
        return self.learn_step(env)
        

    def _min(self, env: TicTacToe):
        board_hash = env.getHash()
        if board_hash in self.cache:
            return self.cache[board_hash]
        min_value = self.DRAW_VALUE
        action = None

        winner = env.who_won()
        if winner == self.side:
            min_value = self.WIN_VALUE
            action = None
        elif winner == env.other_side(self.side):
            min_value = self.LOSS_VALUE
            action = None
        else:
            empty_space = env.getEmptySpaces()
            for space in empty_space:
                new_env = copy.deepcopy(env)
                new_env.step(space)

                res, _ = self._max(new_env)
                if res < min_value or action is None:
                    min_value = res
                    action = space
                    if min_value == self.LOSS_VALUE:
                        self.cache[board_hash] = (min_value, action)
                        return min_value, action

                self.cache[board_hash] = (min_value, action)
        return min_value, action

    def _max(self, env: TicTacToe):
        board_hash = env.getHash()
        if board_hash in self.cache:
            return self.cache[board_hash]

        max_value = self.DRAW_VALUE
        action = None

        winner = env.who_won()
        if winner == self.side:
            max_value = self.WIN_VALUE
            action = None
        elif winner == env.other_side(self.side):
            max_value = self.LOSS_VALUE
            action = None
        else:
            empty_space = env.getEmptySpaces()
            for space in empty_space:
                b = copy.deepcopy(env)
                b.step(space)

                res, _ = self._min(b)
                if res > max_value or action is None:
                    max_value = res
                    action = space

                    if max_value == self.WIN_VALUE:
                        self.cache[board_hash] = (max_value, action)
                        return max_value, action

                self.cache[board_hash] = (max_value, action)
        return max_value, action

    def learn_step(self, env: TicTacToe):
        score, action = self._max(env)
        return env.step(action)

    def update(self, env, reward):
        pass


def move(env, pi, s, actions, random=False, verbos=False):
    if verbos:
        env.printBoard()
    if random:
        a = np.random.randint(len(actions))
        return env.step(actions[a])
    return pi.move(env)


def plot_test_game(env, pi1, pi2, random_crosses=False, random_naughts=True):
    '''Играем тестовую партию между стратегиями или со случайными ходами, рисуем ход игры'''
    done = False
    env.reset()
    while not done:
        s, actions = env.getHash(), env.getEmptySpaces()
        if env.curTurn == 1:
            observation, reward, done, info = move(
                env, pi1, s, actions, random=random_crosses, verbos=True)
        else:
            observation, reward, done, info = move(
                env, pi2, s, actions, random=random_naughts, verbos=True)
        if reward == 1:
            print("Крестики выиграли!")
        if reward == -1:
            print("Нолики выиграли!")
    env.printBoard()


if __name__ == "__main__":
    agent = MinMaxAgent()
    agent.new_game(-1)
    env = TicTacToe(4,4,4)
    plot_test_game(env, None, agent, True, False)
