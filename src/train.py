import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv,self).__init__()
        self.size=10
        self.actionSpace=gym.spaces.Discrete(4)
        self.observationSpace=gym.spaces.Box(low=0,high=1,shape=(self.size,self.size,3),dtype=np.float32)
        self.reset()

    def reset(self):
        self.snake=[[5,5]]
        self.direction=0
        self.food=self._placeFood()
        self.done=False
        return self._getState()

    def step(self,action):
        if abs(action-self.direction)!=2:
            self.direction=action
        head=self.snake[0].copy()
        if self.direction==0: head[1]-=1
        elif self.direction==1: head[0]+=1
        elif self.direction==2: head[1]+=1
        else: head[0]-=1
        reward=0
        if self._isCollision(head):
            self.done=True
            reward=-1
            return self._getState(),reward,self.done,{}
        if head==self.food:
            self.snake.insert(0,head)
            reward=1
            self.food=self._placeFood()
        else:
            self.snake.insert(0,head)
            self.snake.pop()
        return self._getState(),reward,self.done,{}

    def _getState(self):
        state=np.zeros((self.size,self.size,3),dtype=np.float32)
        for x,y in self.snake:
            if 0<=x<self.size and 0<=y<self.size:
                state[y,x,0]=1
        fx,fy=self.food
        state[fy,fx,1]=1
        return state

    def _placeFood(self):
        while True:
            pos=[np.random.randint(0,self.size),np.random.randint(0,self.size)]
            if pos not in self.snake:
                return pos

    def _isCollision(self,head):
        x,y=head
        return x<0 or y<0 or x>=self.size or y>=self.size or head in self.snake[1:]

class DQN(nn.Module):
    def __init__(self,obsShape,nActions):
        super(DQN,self).__init__()
        self.net=nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obsShape),128),
            nn.ReLU(),
            nn.Linear(128,nActions)
        )
    def forward(self,x):
        return self.net(x)

if __name__ == "__main__":
    env=SnakeEnv()
    obsShape=env.observationSpace.shape
    nActions=env.actionSpace.n
    model=DQN(obsShape,nActions)
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    memory=deque(maxlen=10000)
    batchSize=32
    gamma=0.9
    epsilon=1.0
    decay=0.995

    for episode in range(2000):
        state=env.reset()
        totalReward=0
        while True:
            if np.random.rand()<epsilon:
                action=np.random.randint(nActions)
            else:
                with torch.no_grad():
                    q=model(torch.tensor(state).unsqueeze(0))
                    action=torch.argmax(q).item()
            nextState,reward,done,_=env.step(action)
            memory.append((state,action,reward,nextState,done))
            state=nextState
            totalReward+=reward
            if done:
                break
            if len(memory)>batchSize:
                batch=random.sample(memory,batchSize)
                states,actions,rewards,nextStates,dones=zip(*batch)
                states=torch.tensor(np.array(states))
                actions=torch.tensor(actions)
                rewards=torch.tensor(rewards)
                nextStates=torch.tensor(np.array(nextStates))
                dones=torch.tensor(dones)
                qValues=model(states).gather(1,actions.unsqueeze(1)).squeeze(1)
                nextQ=model(nextStates).max(1)[0]
                target=rewards+gamma*nextQ*(1-dones.float())
                loss=nn.functional.mse_loss(qValues,target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epsilon*=decay
        print(f"Episode {episode+1}: reward={totalReward:.2f}, epsilon={epsilon:.3f}")

    torch.save(model.state_dict(),"snake_model.pth")
    print("Model saved as snake_model.pth")