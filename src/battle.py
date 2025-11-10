import torch
import matplotlib.pyplot as plt
from battle_env import SnakeBattleEnv
from train import DQN

device = torch.device("cpu")
env = SnakeBattleEnv(size=200)

obsShape = env.env1.observationSpace.shape
nActions = env.env1.actionSpace.n

model1 = DQN(obsShape[0], nActions)
model2 = DQN(obsShape[0], nActions)

model1.load_state_dict(torch.load("snake_model_final.pth", map_location=device))
model2.load_state_dict(torch.load("snake_model_final.pth", map_location=device))
model1.eval()
model2.eval()

plt.ion()
fig, ax = plt.subplots()

for episode in range(3):
    s1, s2 = env.reset()
    done = False
    total1, total2 = 0, 0

    while not done:
        ax.clear()
        ax.imshow(env.renderBoard())
        plt.title(f"Episode {episode+1} | S1: {total1:.1f} | S2: {total2:.1f}")
        plt.pause(0.05)

        with torch.no_grad():
            q1 = model1(torch.tensor(s1, dtype=torch.float32).unsqueeze(0))
            q2 = model2(torch.tensor(s2, dtype=torch.float32).unsqueeze(0))
            a1 = torch.argmax(q1).item()
            a2 = torch.argmax(q2).item()

        s1, s2, r1, r2, done = env.step(a1, a2)
        total1 += r1
        total2 += r2

    print(f"Episode {episode+1}: Snake1={total1:.1f}, Snake2={total2:.1f}")

plt.ioff()
plt.show()