from typing import Any, List
import gym
from gym import spaces
from gym.spaces.discrete import Discrete
from gym.utils import seeding
from numpy import random


def cmp(a, b):
    return float(a > b) - float(a < b)


MIN_CARDS = 15

COUNV_DICT = {
    2:1, 3:2, 4:2, 5:3,
    6:2, 7:1, 8:0, 9:-1,
    10:-2, 1:-2
}

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackDoubleCountEnv(gym.Env):
    def __init__(self, natural=False, decks_number=4):
        self.decks_number = decks_number
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(22*self.decks_number+1)) # [-11,11]*количество колод+0
        )
        self.seed()
        # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
        self.deck = deck[:]*self.decks_number
        self.count = 11*self.decks_number  # сдвиг 0 отностительно минимального значения

        self.natural = natural
        self.dealer = []
        self.player = []
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_card(self):
        card = int(self.np_random.choice(self.deck))
        self.deck.remove(card)
        self.count += COUNV_DICT[card]
        return card

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            card = self.draw_card()
            self.player.append(card)
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        elif action == 0:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                card = self.draw_card()
                self.dealer.append(card)
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
        elif action == 2:
            card = self.draw_card()
            self.player.append(card)
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = cmp(score(self.player), score(self.dealer))*2
        else:
            raise ValueError('invalid action')
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        assert 0 <= self.count <= 22*self.decks_number+1
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), self.count)

    def reset(self):
        if len(self.deck) < MIN_CARDS:
            self.deck = deck[:]*self.decks_number
            self.count = 11*self.decks_number  # сдвиг 0 отностительно минимального значения
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        return self._get_obs()