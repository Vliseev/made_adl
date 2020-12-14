import copy
import math
import random
import numpy as np
from typing import List

from tic_tac_toe import TicTacToe

class Node(object):
    def __init__(self, env: TicTacToe, parent=None, move=None) -> None:
        self.env = env
        self.isTerminal = not env.isTerminal() is None
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.move = move
        self.numVisits = 0
        self.totalReward = 0
        self.win_counts = {
            1: 0,
            -1: 0,
        }
        self.children: List[Node] = []
        self.unvisited_moves = list(env.getEmptySpaces())

    def add_rand_child(self, rnd):
        idx = rnd.randint(0, len(self.unvisited_moves)-1)
        action = self.unvisited_moves[idx]
        self.unvisited_moves.pop(idx)
        node_env = copy.deepcopy(self.env)
        node_env.step(action)
        new_node = Node(node_env, self, action)
        self.children.append(new_node)
        return new_node

    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def get_winning_perc(self, player):
        return float(self.win_counts[player])/float(self.numVisits)


class MCTSAgent(object):
    def __init__(self, num_rounds=10_000, expl_factor=1 / math.sqrt(2), seed=123):
        self.num_rounds = num_rounds
        self.expl_factor = expl_factor
        self.rnd = random.Random(seed)

    def get_best_move(self, root, env):
        best_move = None
        best_pct = -1.0
        for child in root.children:
            win_perc = child.get_winning_perc(-env.curTurn)
            if win_perc > best_pct:
                best_pct = win_perc
                best_move = child.move
        return best_move

    def get_action(self, env):
        root = Node(env)
        for n_round in range(self.num_rounds):
            node: Node = root
            while not node.isTerminal and not node.can_add_child():
                node = self.select_child(node, env)
            if node.can_add_child():
                node = node.add_rand_child(self.rnd)
            reward, winner = self.random_game(copy.deepcopy(env))
            self.propagate(node, reward, winner)
        return self.get_best_move(root, env)

    def select_child(self, node: Node, env):
        visit_sum = sum(child.numVisits for child in node.children)
        visit_sum = math.log(visit_sum)
        max_score, best_chld = -1, None
        for child in node.children:
            win_perc = node.get_winning_perc(-env.curTurn)
            exp_fact = math.sqrt(visit_sum/child.numVisits)
            score = win_perc + self.expl_factor*exp_fact
            if score > max_score:
                max_score, best_chld = score, child
        return best_chld

    def propagate(self, node: Node, reward, winner):
        while node is not None:
            node.numVisits += 1
            if reward == 0:
                node.win_counts[1] += 0.5
                node.win_counts[-1] += 0.5
            else:
                node.win_counts[winner] += reward
            node = node.parent

    def random_game(self, env: TicTacToe):
        done = not env.isTerminal() is None
        done = False if done is None else done
        reward = None
        while not done:
            actions = env.getEmptySpaces()

            cur_action = self.rnd.choice(actions)
            _, reward, done, _ = env.step(cur_action)
        return reward, env.who_won()
