import numpy as np
from tic_tac_toe import TicTacToe, TikTakCounter
from collections import namedtuple
import tqdm
import random
import torch
from torch import nn
import math
from torch import optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.rng = random.Random(seed)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return self.rng.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN1(nn.Module):
    def __init__(self, w=3, h=3, n_outputs=9):
        super(DQN1, self).__init__()
        self.w = w
        self.h = h

        def conv2d_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        w = conv2d_size_out(w, 3)
        h = conv2d_size_out(h, 3)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(w*h*32, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_outputs)

    def forward(self, x):
        """Forward pass for the model.
        """
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def get_action(self, state):
        with torch.no_grad():
            ans = self.forward(state)
            return ans.max(1)[1].view(1, 1)


class DQNTrainer(object):
    WIN = 1.0
    DRAW = 0.5
    LOSS = 0.0

    def __init__(self, env: TicTacToe, alpha=0.05,
                 gamma=1, seed=123,
                 memory_size=60_000, batch_size=64):
        self.side = None
        self.n_states = env.n_cols * env.n_rows
        self.alpha, self.gamma = alpha, gamma
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
        self.h, self.w = env.n_cols, env.n_cols
        self.nn_init(env)
        self.memory = ReplayMemory(memory_size, seed)
        self.num_step = 0
        self.eps_init, self.eps_final, self.eps_decay = 0.9, 0.05, 200

    def nn_init(self, env):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DQN1(env.n_cols, env.n_rows,
                           env.n_cols*env.n_rows).to(self.device)
        self.target = DQN1(env.n_cols, env.n_rows,
                           env.n_cols*env.n_rows).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=2e-4)

    def get_next_step(self, state):
        self.num_step += 1
        eps_threshold = self.eps_final + \
            (self.eps_init - self.eps_final) * \
            math.exp(-1. * self.num_step / self.eps_decay)
        if self.rnd.rand() > 1-eps_threshold:
            return self.target.get_action(state)
        else:
            return torch.tensor(
                [[random.randrange(0, self.w*self.h)]],
                device=self.device,
                dtype=torch.long,
            )

    def train(self, side, env: TicTacToe,  n_steps):
        counter = TikTakCounter()

        self.side = side
        env.reset()
        self.num_step = 0
        state = torch.tensor(env.board, dtype=torch.float,
                             device=self.device).view(1, 1, self.w, self.h)
        for n_step in tqdm.tqdm(range(n_steps)):
            action = self.get_next_step(state)
            _, reward, done, _ = env.step(env.action_from_int(action))
            next_state = None
            if done:
                self.update_counter(reward, counter)
                next_state = None
            if not done:
                _, reward_next, done, _ = env.step(
                    env.action_from_int(
                        self.get_rand_action(env)
                    ))
                next_state = torch.tensor(
                    env.board, dtype=torch.float, device=self.device).view(1, 1, self.w, self.h)
                if done:
                    self.update_counter(reward_next, counter)
                    next_state = None

            reward = torch.tensor(
                [reward], dtype=torch.float, device=self.device)

            self.memory.push(state, action, next_state, reward)
            state = next_state
            if done:
                env.reset()
                state = torch.tensor(
                    env.board, dtype=torch.float, device=self.device).view(1, 1, self.w, self.h)
            if self.num_step % 5_000 == 0:
                """
                Мы играем сетью target, а учим policy, каждые 5_000 шагов обновляем ее
                """
                self.target.load_state_dict(self.policy.state_dict())
                print(counter)
            self.learn()
        return counter

    def update_counter(self, reward, counter: TikTakCounter):
        if reward > 0:
            counter.cross += 1
        elif reward == -1:
            counter.draw += 1
        elif reward == 0:
            counter.naughts += 1
        else:
            counter.invalid += 1
        counter.tot += 1
        counter.update_history()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(
            *transitions)
        non_final_mask = torch.tensor(
            [not s is None for s in batch_next_state],
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch_next_state if s is not None])
        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        target_act = self.target(non_final_next_states)
        next_state_values[non_final_mask] = target_act.max(1)[0].detach()
        state_action_values = self.policy(batch_state).gather(1, batch_action)

        expected_state_action_values = (
            next_state_values * self.gamma) + batch_reward

        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def get_rand_action(self, env):
        actions = env.getEmptySpaces()
        return self.rnd.randint(0, len(actions))


if __name__ == "__main__":
    env = TicTacToe(5, 5, 3)
    trainer = DQNTrainer(env, memory_size=5_000, batch_size=512)
    trainer.train(1, env, 100_000)
