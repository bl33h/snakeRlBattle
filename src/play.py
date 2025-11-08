import torch
import numpy as np
import matplotlib.pyplot as plt
from train import SnakeEnv, DQN

env=SnakeEnv()
obsShape=env.observationSpace.shape
nActions=env.actionSpace.n
model=DQN(obsShape,nActions)
model.load_state_dict(torch.load("snake_model.pth"))
model.eval()

plt.ion()
fig,ax=plt.subplots()

for episode in range(5):
    state=env.reset()
    totalReward=0
    while True:
        ax.clear()
        ax.imshow(env._getState())
        plt.title(f"Trained Snake | Episode {episode+1} | Reward: {totalReward}")
        plt.pause(0.1)
        with torch.no_grad():
            q=model(torch.tensor(state).unsqueeze(0))
            action=torch.argmax(q).item()
        nextState,reward,done,_=env.step(action)
        state=nextState
        totalReward+=reward
        if done:
            break
    print(f"Episode {episode+1} finished with reward {totalReward}")

plt.ioff()
plt.show()