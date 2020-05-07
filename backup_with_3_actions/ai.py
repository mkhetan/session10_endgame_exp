# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Actor(nn.Module):

    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()

#        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 40

#        self.conv1 = nn.Sequential(
#                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
#                     nn.ReLU(), nn.BatchNorm2d(16)
#                     )  # output = 38

#        self.conv2 = nn.Sequential(
#            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
#            nn.ReLU(), nn.BatchNorm2d(16)
#        )  # output = 36

#        self.conv3 = nn.Sequential(
#            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
#            nn.ReLU(), nn.BatchNorm2d(16)
#        )  # output = 34

#        self.conv4 = nn.Sequential(
#            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
#            nn.ReLU(), nn.BatchNorm2d(16)
#        )  # output = 32

#        self.conv5 = nn.Sequential(
#            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
#            nn.ReLU(), nn.BatchNorm2d(16)
#        )  # output = 30

#        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 15

#        self.conv6 = nn.Sequential(
#            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
#            nn.ReLU(), nn.BatchNorm2d(16)
#        )  # output = 13

#        self.conv7 = nn.Sequential(
#            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
#            nn.ReLU(), nn.BatchNorm2d(16)
#        )  # output = 11

#        self.conv8 = nn.Sequential(
#            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
#            nn.ReLU(), nn.BatchNorm2d(16)
#        )  # output = 9

#        self.conv9 = nn.Sequential(
#            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
#            nn.ReLU(), nn.BatchNorm2d(16)
#        )  # output = 7

#        self.conv10 = nn.Sequential(
#            nn.Conv2d(16, action_dim, kernel_size=(7, 7), padding=0, bias=False),
#            # nn.ReLU(), nn.BatchNorm2d(16)
#        )  # output = 1
        # the copied network
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(80)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(80)))
        linear_input_size = convw * convh * 32
        self.linear_1 = nn.Linear(linear_input_size, action_dim)

        #self.layer_1 = nn.Linear(state_dim, 400)
        #self.layer_2 = nn.Linear(400, 300)
        #self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
#        x = self.pool1(x)
#        x = self.conv1(x)
#        x = self.conv2(x)
#        x = self.conv3(x)
#        x = self.conv4(x)
#        x = self.conv5(x)
#        x = self.pool2(x)
#        x = self.conv6(x)
#        x = self.conv7(x)
#        x = self.conv8(x)
#        x = self.conv9(x)
#        x = self.conv10(x)
#        x = x.view(x.size(0), -1)
#        x = self.max_action * torch.tanh(x)
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.shape)
        x = self.linear_1(x.view(x.size(0), -1))
        x = self.max_action * torch.tanh(x)
        return x


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(80)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(80)))
        linear_input_size = convw * convh * 32
#        self.linear_1 = nn.Linear(linear_input_size, action_dim)

        self.layer_1 = nn.Linear(linear_input_size + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network

        self.conv4 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        convw2 = conv2d_size_out(conv2d_size_out(conv2d_size_out(80)))
        convh2 = conv2d_size_out(conv2d_size_out(conv2d_size_out(80)))
        linear_input_size2 = convw2 * convh2 * 32

        self.layer_4 = nn.Linear( linear_input_size2 + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = x1.view(x1.size(0), -1)
        xu = torch.cat([x1, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)

        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x2 = F.relu(self.bn3(self.conv3(x2)))
        x2 = x2.view(x2.size(0), -1)
        xu = torch.cat([x2, u], 1)
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = x1.view(x1.size(0), -1)
        xu = torch.cat([x1, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

# Implementing Experience Replay

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
    
    def __init__(self, state_dim, action_dim, max_action):
        #self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor = Actor(action_dim, max_action).to(device)
        self.actor_target = Actor(action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1,1,80,80)).to(device)
        #state = torch.Tensor(state).to(device)
        #print(state.shape)
        #print(state)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            #print("next_action_shape")
            #print(next_action.shape)
            #print(next_state.shape)
            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            #target_Q1, target_Q2 = self.critic_target(next_state.reshape(batch_size, 80*80), next_action.reshape(batch_size, 1))
            #target_Q1, target_Q2 = self.critic_target(next_state.reshape(batch_size, 80*80), next_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            #current_Q1, current_Q2 = self.critic(state, action)
            #current_Q1, current_Q2 = self.critic(state.reshape(batch_size, 80*80), action.reshape(batch_size, 1))
            #current_Q1, current_Q2 = self.critic(state.reshape(batch_size, 80*80), action)
            current_Q1, current_Q2 = self.critic(state, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                #actor_loss = -self.critic.Q1(state.reshape(batch_size, 80*80), self.actor(state).reshape(batch_size,1)).mean()
                #actor_loss = -self.critic.Q1(state.reshape(batch_size, 80*80), self.actor(state)).mean()
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

