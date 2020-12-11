import numpy as np
import gym
from numpy.core.numeric import cross
from tic_tac_toe import TicTacToe
from collections import defaultdict
import tqdm


class TikTakCounter:
    def __init__(self) -> None:
        self.cross = 0
        self.naughts = 0
        self.draw = 0
        self.tot = 0
        self.invalid = 0

    def __str__(self) -> str:
        return "cross={} naughts={} tot={} draw={} invalid={}".format(
            self.cross, self.naughts, self.tot, self.draw, self.invalid)

    def __repr__(self) -> str:
        return self.__str__()


class QleaningAgent(object):
    WIN = 1.0
    DRAW = 0.5
    LOSS = 0.0

    def __init__(self, env: TicTacToe, alpha=0.05,
                 epsilon=0.0, gamma=1, seed=123):
        self.side = None
        self.n_states = env.n_cols * env.n_rows
        self.Q = defaultdict(lambda: np.zeros(self.n_states))
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.rnd = np.random.RandomState(seed)
        self.actions_history = []
        self.pi = {}

    def get_next_step(self, state, env, train=False):
        use_pi = state in self.pi
        if train:
            use_pi = use_pi and self.rnd.rand() > self.epsilon
        if use_pi:
            return self.pi[state]
        else:
            action = self.rnd.randint(0, self.n_states)
            return action

    def update_pi(self):
        for key in self.Q:
            max_rew_idx = np.argmax(self.Q[key])
            self.pi[key] = max_rew_idx

    def learn_step(self, env: TicTacToe):
        state = env.getHash()
        action = self.get_next_step(state, env, True)
        self.actions_history.append((state, action))
        return env.step(env.action_from_int(action))

    def update(self, env: TicTacToe, reward):
        self.actions_history.reverse()
        if self.side == env.who_won():
            reward = abs(reward)
        elif env.other_side(self.side) == env.who_won():
            reward = -abs(reward)
        elif env.who_won() == 0:
            reward = self.DRAW
        else:
            raise RuntimeError("invalid {} {}".format(
                self.side, env.who_won()))

        max_val = -float("inf")
        for state, action in self.actions_history:
            q = self.Q[state]
            if max_val < 0:
                q[action] = reward
            else:
                q[action] = q[action] + self.alpha * \
                    (self.gamma*max_val - q[action])
            max_val = np.max(q)
        self.update_pi()

    def new_game(self, side):
        self.side = side
        self.actions_history = []


def move(env, pi, actions, random=False, verbos=False):
    if verbos:
        env.printBoard()
    if random:
        a = np.random.randint(len(actions))
        return env.step(actions[a])
    return pi.learn_step(env)


def plot_test_game(env, agent1, agent2, counter: TikTakCounter, random_crosses=False, random_naughts=True, verbose=False):
    done = False
    env.reset()
    if agent1:
        agent1.new_game(1)
    if agent2:
        agent2.new_game(-1)
    reward = 0
    while not done:
        actions = env.getEmptySpaces()
        if env.curTurn == 1:
            _, reward, done, _ = move(
                env, agent1, actions, random=random_crosses, verbos=verbose
            )
        else:
            _, reward, done, _ = move(
                env, agent2, actions, random=random_naughts, verbos=verbose
            )
        if reward == 1:
            counter.cross += 1
            if verbose:
                print("Крестики выиграли!")
        if reward == -1:
            counter.naughts += 1
            if verbose:
                print("Нолики выиграли!")
        if reward < -1:
            counter.invalid += 1
            if verbose:
                print("Ошибка!")
    if reward == 0:
        counter.draw += 1
        if verbose:
            print("Ничья!")

    if agent1:
        agent1.update(env, reward)
    if agent2:
        agent2.update(env, reward)


if __name__ == "__main__":
    counter = TikTakCounter()
    env = TicTacToe(3, 3, 3)
    agent1 = QleaningAgent(env)
    agent2 = QleaningAgent(env)
    for i in tqdm.tqdm(range(10000)):
        plot_test_game(env, agent1, None, counter, False, True, verbose=False)
        counter.tot += 1
    print(counter)
    # counter = TikTakCounter()
    # for i in tqdm.tqdm(range(10000)):
    #     plot_test_game(env, agent1, agent2, counter,
    #                    False, False, verbose=False)
    #     counter.tot += 1
    # print(counter)
