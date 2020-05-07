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

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        #self.fc1 = nn.Linear(input_size, 30)
        #self.fc2 = nn.Linear(30, nb_action)
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 40

        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
                     nn.ReLU(), nn.BatchNorm2d(16)
                     )  # output = 38

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(), nn.BatchNorm2d(16)
        )  # output = 36

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(), nn.BatchNorm2d(16)
        )  # output = 34

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(), nn.BatchNorm2d(16)
        )  # output = 32

        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(), nn.BatchNorm2d(16)
        )  # output = 30

        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 15

        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(), nn.BatchNorm2d(16)
        )  # output = 13

        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(), nn.BatchNorm2d(16)
        )  # output = 11

        self.conv8 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(), nn.BatchNorm2d(16)
        )  # output = 9

        self.conv9 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(), nn.BatchNorm2d(16)
        )  # output = 7

        self.conv10 = nn.Sequential(
            nn.Conv2d(16, nb_action, kernel_size=(7, 7), padding=0, bias=False),
            # nn.ReLU(), nn.BatchNorm2d(16)
        )  # output = 3

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
        self.linear_1 = nn.Linear(linear_input_size, nb_action)

    def forward(self, x):
#        x = F.relu(self.fc1(state))
#        q_values = self.fc2(x)
#        return q_values
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
#        return x
# print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
# print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
# print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
# print(x.shape)
        x = self.linear_1(x.view(x.size(0), -1))
#       x = self.max_action * torch.tanh(x)
        return x


# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        tempT = np.ones((80, 80))
        self.last_state = torch.Tensor(tempT.reshape((1, 80, 80))).float().unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")