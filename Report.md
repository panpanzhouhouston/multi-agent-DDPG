[//]: # (Paper References)

[Paper1]: https://arxiv.org/pdf/1810.02912.pdf
[image2]: https://arxiv.org/pdf/1805.08776.pdf

# Multi-agent DDPG Report

### Introduction

For this project, multiple AI agent is trained to control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play. After the training, the average score over 100 eposide can reach above 1.5.


### Algorithm
In this project, we use a centralized critic/decentralized actor multi-agent reinforcement learning algorithm introduced in [here](https://arxiv.org/pdf/1810.02912.pdf). Each agent has a policy network to generate the action. All agents have a shared value network that generate the values of all agents state and action. The detailed algorithm is introduced as below two figures.

![results](method.jpg)
![results](algorithm.jpeg)

### Model Architechture

- **_Actor Network_**
    - Hidden: (state_size * agent_num, 256) - ReLU
    - Hidden: (256, 128) - ReLU
    - Output: (128, action_size) - TanH

- **_Critic Network_**
    - Hidden: (state_size * agent_num, 256) - ReLU
    - Hidden: (256 + action_size * agent_num, 128) - ReLU
    - Hidden: (128, 64) - ReLU
    - Output: (64, 1) - Linear

### Hyperparameters
 - BUFFER_SIZE = int(1e6)  # replay buffer size
 - BATCH_SIZE = 128        # minibatch size
 - GAMMA = 0.99            # discount factor
 - TAU = 1e-3              # for soft update of target parameters
 - device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # the network is run on Nvida GPU  1080

- **_Initial_Search_**
 - LR_ACTOR = 1e-4        # learning rate of the actor 
 - LR_CRITIC = 1e-4        # learning rate of the critic
 - LR_DECAY = 1         # In each episode, the learning rate is decayed by a factor
 - NOISE_DECAY_FACTOR = 1         # noise decay factor after each sampling
 - sigma=0.5           # noise intensity in OUNoise
 
 - **_Optimizing_**
 - LR_ACTOR = 1e-5        # learning rate of the actor 
 - LR_CRITIC = 1e-5        # learning rate of the critic
 - LR_DECAY = 1         # In each episode, the learning rate is decayed by a factor
 - NOISE_DECAY_FACTOR = 1         # noise decay factor after each sampling
 - sigma=0.1            # noise intensity in OUNoise
 
  - **_Finalizing_**
 - LR_ACTOR = 1e-6        # learning rate of the actor 
 - LR_CRITIC = 1e-6        # learning rate of the critic
 - LR_DECAY = 1         # In each episode, the learning rate is decayed by a factor
 - NOISE_DECAY_FACTOR = 0.99         # noise decay factor after each sampling
 - sigma=0.01            # noise intensity in OUNoise


### Results

Initial searching process, very slow and not stable.

![results](search.png)

After the network loss gradient had reasonable trend, smaller noise and learning rate is used to gradually optimize the weight.

![results](finalize.png)

Finally, very small noise and lr is used to reach the best performace. The average score is kept above 0.5 for more than 2000 episode and the highest average score reaches 1.5.

![results](super_performance.png)

![results](Tennis.gif)

### Future Work
   - 1. Try some different actor/critic network artitechture to test the performance.

   - 2. Search approprate learning rate and noise decay rate to optimize the training process.

   - 3. A more difficult **Soccer** environment worth to explore to seek how to train agents to collabrate and competition. [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos).  
