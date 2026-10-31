"""DDPG agent for environments with many parallel agents (e.g. the 20-arm Reacher).

All arms share ONE actor, ONE critic and ONE replay buffer. Each arm only
adds its own experience to the shared buffer, so 20 arms collect 20x more
data per environment step than a single arm.

Differences from ddpg_agent.py:
    1. act() takes a batch of states (one row per arm) and returns a batch of actions.
    2. step() adds the transitions of ALL arms, but only learns every LEARN_EVERY
       steps (LEARN_TIMES updates each time) instead of after every transition.
    3. The OU noise has one independent process per arm, so arms explore differently.
    4. The OU noise uses Gaussian samples (the original used uniform [0, 1) samples,
       which are always positive and push every action in the same direction).
    5. The critic's gradients are clipped to keep training stable.
    6. The replay buffer stores experiences in fixed float32 arrays (~290 MB for 1e6),
       instead of a deque of Python tuples (1.5 GB+), so memory can't run out mid-training.
"""

import numpy as np
import random
import copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size (larger, since 20 arms fill it fast)
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20        # learn once every LEARN_EVERY environment steps...
LEARN_TIMES = 10        # ...and run LEARN_TIMES updates each time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent():
    """Shared DDPG agent that controls several arms at once."""

    def __init__(self, state_size, action_size, num_agents, random_seed, noise_sigma=0.2):
        """Initialize a MultiAgent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of parallel arms in the environment
            random_seed (int): random seed
            noise_sigma (float): size of the exploration noise (smaller = less random actions)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network) - shared by all arms
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network) - shared by all arms
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Start the target networks as exact copies of the local networks
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # Noise process: shape (num_agents, action_size) -> one independent process per arm
        self.noise = OUNoise((num_agents, action_size), random_seed, sigma=noise_sigma)

        # Replay memory - shared by all arms
        self.memory = ReplayBuffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Counts environment steps, to know when to learn
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save the experience of every arm, then learn every LEARN_EVERY steps.

        Params
        ======
            states (array): shape (num_agents, state_size)
            actions (array): shape (num_agents, action_size)
            rewards (list): one reward per arm
            next_states (array): shape (num_agents, state_size)
            dones (list): one done flag per arm
        """
        # Save one experience per arm in the shared buffer (all 20 rows in one call)
        self.memory.add_batch(states, actions, rewards, next_states, dones)

        # Learn only every LEARN_EVERY steps, and only once the buffer has enough samples.
        # Learning after every transition would mean 20 updates per step - very slow.
        self.t_step = (self.t_step + 1) % LEARN_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(LEARN_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Return one action per arm for the given states, using the current policy.

        Params
        ======
            states (array): shape (num_agents, state_size)
            add_noise (bool): add exploration noise (set False when evaluating)
        """
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            # The actor handles the whole batch at once -> shape (num_agents, action_size)
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            # Each row of the noise belongs to one arm
            actions += self.noise.sample()
        # The environment only accepts actions in [-1, 1]
        return np.clip(actions, -1, 1)

    def reset(self):
        """Reset the noise of every arm (call at the start of each episode)."""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip the critic's gradients so one bad batch can't make a huge update
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss: the actor wants actions the critic scores highly
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        """Copy the local model's weights into the target model exactly."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save(self, path, **extra):
        """Save everything needed to continue training later.

        Saves the local AND target networks plus both optimizers, so training
        resumes exactly where it stopped. The replay buffer is not saved
        (it would be huge); it refills within a few episodes.

        Params
        ======
            path (str): file to write, e.g. 'checkpoint_multi.pth'
            extra: anything else to store, e.g. scores=[...]
        """
        # numpy numbers can't be read back by torch.load's safe mode -> store plain floats
        if 'scores' in extra:
            extra['scores'] = [float(s) for s in extra['scores']]
        checkpoint = {
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        checkpoint.update(extra)
        torch.save(checkpoint, path)

    def load(self, path):
        """Load a checkpoint written by save() and return it.

        The returned dict also holds the extras passed to save(),
        e.g. checkpoint['episode'] and checkpoint['scores'].
        """
        checkpoint = torch.load(path, map_location=device)
        self.actor_local.load_state_dict(checkpoint['actor_local'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_local.load_state_dict(checkpoint['critic_local'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        return checkpoint

    def load_weights(self, actor_path, critic_path):
        """Load plain actor/critic weights (e.g. checkpoint_actor_multi.pth).

        Use this for files saved with torch.save(model.state_dict(), ...).
        Only the local networks were saved, so the targets start as copies of them
        and the optimizers start fresh.
        """
        self.actor_local.load_state_dict(torch.load(actor_path, map_location=device))
        self.critic_local.load_state_dict(torch.load(critic_path, map_location=device))
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

class OUNoise:
    """Ornstein-Uhlenbeck process, with one independent process per arm."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.

        Params
        ======
            size (tuple): shape of the noise, e.g. (num_agents, action_size)
            seed (int): random seed
            mu (float): long-run mean the noise is pulled towards
            theta (float): how strongly the noise is pulled back to mu
            sigma (float): size of the random kicks
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # Gaussian kicks (mean 0), drawn independently for every arm and every action
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer of experiences, stored in pre-allocated float32 arrays.

    Why arrays instead of a deque of tuples: a tuple of small numpy arrays costs
    several hundred bytes of Python overhead per experience (over 1.5 GB for 1e6
    experiences). Plain float32 arrays need only ~290 MB for the same 1e6, and
    sampling is a single fancy-index instead of stacking 256 Python objects.
    """

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # one row per experience; fixed size, so the buffer can never use more than ~290 MB
        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

        self.pos = 0     # next row to write
        self.size = 0    # how many rows hold real data

    def add_batch(self, states, actions, rewards, next_states, dones):
        """Add one experience per arm (each argument has one row per arm)."""
        n = len(rewards)
        # rows to write; wrap around to the start when the buffer is full (oldest data is overwritten)
        idx = (self.pos + np.arange(n)) % self.buffer_size
        self.states[idx] = states
        self.actions[idx] = actions
        self.rewards[idx, 0] = rewards
        self.next_states[idx] = next_states
        self.dones[idx, 0] = dones
        self.pos = (self.pos + n) % self.buffer_size
        self.size = min(self.size + n, self.buffer_size)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        idx = self.rng.integers(0, self.size, size=self.batch_size)

        states = torch.from_numpy(self.states[idx]).to(device)
        actions = torch.from_numpy(self.actions[idx]).to(device)
        rewards = torch.from_numpy(self.rewards[idx]).to(device)
        next_states = torch.from_numpy(self.next_states[idx]).to(device)
        dones = torch.from_numpy(self.dones[idx]).to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size
