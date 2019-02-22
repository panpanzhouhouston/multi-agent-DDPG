import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5        # learning rate of the actor
LR_CRITIC = 1e-5        # learning rate of the critic
LR_DECAY = 1
NOISE_DECAY_FACTOR = 1         # noise decay factor after each sampling
UPDATE_EVERY_N_STEPS = 4
CRITIC_WEIGHT_PREFIX = 'critic_parameters'
ACTOR_WEIGHT_PREFIX = 'actor_parameters_agent_'
AGENT_TRAINING = [0, 1]

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class MultiAgent():
    """Multiple agent interact and learn at the same time"""
    
    def __init__(self, state_size, action_size, agent_num, random_seed, use_actor_weight=False, use_critic_weight=False):
        """
        Initilize the multi-agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_num = agent_num
        self.seed = random.seed(random_seed)
        
        # Critic Network (w/ Target Network) This is a shared network
        self.critic_local = Critic(state_size, action_size, agent_num, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, agent_num, random_seed).to(device)
        self.critic_weight_name = CRITIC_WEIGHT_PREFIX + '.pth'


        # Load the critic_weight if exist
        if use_critic_weight:
            try:
                self.critic_local.load_state_dict(torch.load(self.critic_weight_name))
                self.critic_target.load_state_dict(torch.load(self.critic_weight_name))

            except:
                print('Critic-Net weight loading failed, use random instead')
        
        # initialize buffer for memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        self.agents = [Agent(state_size, action_size, agent_num, random_seed, self.critic_local, 
                             self.critic_target, self.memory, i, use_actor_weight) for i in range(agent_num)]

    def step(self, states, actions, rewards, next_states, dones):

        experience = zip(states, actions, rewards, next_states, dones)
        for i, e in enumerate(experience):
            state, action, reward, next_state, done = e
            id_others = [j for j in range(self.agent_num) if j != i]
            others_states = [states[id_other] for id_other in id_others]
            others_actions = [actions[id_other] for id_other in id_others]
            others_next_states = [next_states[id_other] for id_other in id_others]
            if i in AGENT_TRAINING:
                self.agents[i].step(state, others_states, action, others_actions, reward, next_state,
                                    others_next_states, done, True)
            else:
                self.agents[i].step(state, others_states, action, others_actions, reward, next_state,
                                    others_next_states, done, False)

    def act(self, states, add_noise=True):
        actions = np.zeros((self.agent_num, self.action_size))
        for idx, agent in enumerate(self.agents):
            actions[idx,:] = agent.act(states[idx], add_noise)
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def lr_step(self):
        for i, agent in enumerate(self.agents):
            agent.actor_lr_step()
            if i == 0:
                agent.critic_lr_step()

    def save_weight(self):
        torch.save(self.critic_local.state_dict(), self.critic_weight_name)
        for agent in self.agents:
            agent.save_weight()

class Agent():

    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, agent_num, random_seed, critic_local, critic_target, memory, agent_i, use_actor_weight=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.agent_num = agent_num
        self.memory = memory

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma = LR_DECAY)
        self.agent_weight_name = ACTOR_WEIGHT_PREFIX + str(agent_i) + '.pth'

        # Load the actor weight if exist
        if use_actor_weight:
            try:
                self.actor_local.load_state_dict(torch.load(self.agent_weight_name))
                self.actor_target.load_state_dict(torch.load(self.agent_weight_name))

            except:
                print('Actor-Net weight loading failed, use random instead')

        # critic network optimizer
        self.critic_local = critic_local
        self.critic_target = critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=LR_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed, NOISE_DECAY_FACTOR)

        # Initialize time step
        self.t_step = 0

    def actor_lr_step(self):
        """Schedule the learning rate, decay factor 0.99"""
        self.actor_scheduler.step()

    def critic_lr_step(self):
        self.critic_scheduler.step()

    def step(self, state, others_states, action, others_actions, reward, next_state, others_next_states, done, train=True):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, others_states, action, others_actions, reward, next_state, others_next_states, done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % UPDATE_EVERY_N_STEPS
        if train and self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
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
        states, others_states, actions, others_actions, rewards, next_states, others_next_states, dones = experiences

        all_states = torch.cat((states, others_states.view(-1, self.state_size * (self.agent_num-1))), dim=1).to(device)
        all_actions = torch.cat((actions, others_actions.view(-1, self.action_size * (self.agent_num-1))), dim=1).to(device)
        all_next_states = torch.cat((next_states, others_next_states.view(-1, self.state_size * (self.agent_num-1))), dim=1).to(device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        all_next_actions = self.actor_target(next_states)
        others_next_states = others_next_states.view(-1, (self.agent_num-1) * self.state_size)
        for idx in range(self.agent_num-1):
            s = others_next_states[:,idx*self.state_size:(idx+1)*self.state_size]
            all_next_actions = torch.cat((all_next_actions, self.actor_target(s)), dim=1)

        all_next_actions = all_next_actions.to(device)


        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        #print('critic_loss:', critic_loss)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #

        # Compute actor loss
        this_actions_pred = self.actor_local(states)

        others_states = others_states.view(-1, (self.agent_num-1)*self.state_size)
        for idx in range(self.agent_num-1):
            s = others_states[:,idx*self.state_size:(idx+1)*self.state_size]
            temp_pred = self.actor_target(s)
            if idx == 0:
                other_actions_pred = temp_pred
            else:
                other_actions_pred = torch.cat((other_actions_pred, temp_pred), dim=1)

        all_actions_pred = torch.cat((this_actions_pred, other_actions_pred), dim=1).to(device)
        actor_loss = -self.critic_local(all_states, all_actions_pred).mean()
        #print('actor loss:', actor_loss)

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

    def save_weight(self):
        torch.save(self.actor_local.state_dict(), self.agent_weight_name)


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    # theta = 0.15, sigma = 0.2
    def __init__(self, size, seed, decay_factor, mu=0., theta=0.15, sigma=0.05):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.decay_factor = decay_factor
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        #self.sigma = self.sigma * self.decay_factor
        #self.theta = self.theta * self.decay_factor
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "others_state", "action", "others_action",
                                                                "reward", "next_state", "others_next_state", "done"])
        self.seed = random.seed(seed)


    def add(self, state, others_state, action, others_action, reward, next_state, others_next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, others_state, action, others_action, reward, next_state, others_next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        others_states = torch.from_numpy(np.vstack([e.others_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        others_actions = torch.from_numpy(np.vstack([e.others_action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        others_next_states = torch.from_numpy(np.vstack([e.others_next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, others_states, actions, others_actions, rewards, next_states, others_next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
