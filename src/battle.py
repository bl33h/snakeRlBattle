import torch
import matplotlib.pyplot as plt
from battle_env import SnakeBattleEnv
from train import DQN
import numpy as np

device = torch.device("cpu")
env = SnakeBattleEnv(size=400)

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

numEpisodes = 3  # number of matches to play
maxSteps = 2000  # prevent infinite games

# for metrics
episode_rewards1 = []
episode_rewards2 = []
episode_steps = []
episode_winners = []   # 1 = gana snake1, 2 = gana snake2, 0 = empate

def moving_avg(x, k=10):
    if len(x) == 0:
        return 0.0
    return float(np.mean(x[-k:]))

for episode in range(numEpisodes):
    s1, s2 = env.reset()
    done = False
    total1, total2 = 0.0, 0.0
    steps = 0

    while not done and steps < maxSteps:
        ax.clear()
        ax.imshow(env.renderBoard())
        plt.title(f"Episode {episode + 1} | S1: {total1:.1f} | S2: {total2:.1f}")
        plt.pause(0.001)

        with torch.no_grad():
            q1 = model1(torch.tensor(s1, dtype=torch.float32).unsqueeze(0))
            q2 = model2(torch.tensor(s2, dtype=torch.float32).unsqueeze(0))
            a1 = torch.argmax(q1).item()
            a2 = torch.argmax(q2).item()

        s1, s2, r1, r2, done = env.step(a1, a2)
        total1 += r1
        total2 += r2
        steps += 1

    # ---- SAVE EPISODE METRICS ----
    episode_rewards1.append(total1)
    episode_rewards2.append(total2)
    episode_steps.append(steps)

    if total1 > total2:
        winner = 1
    elif total2 > total1:
        winner = 2
    else:
        winner = 0
    episode_winners.append(winner)

    # moving averages
    avg_r1 = moving_avg(episode_rewards1, k=10)
    avg_r2 = moving_avg(episode_rewards2, k=10)
    avg_len = moving_avg(episode_steps, k=10)
    win_rate1 = np.mean(np.array(episode_winners) == 1)
    win_rate2 = np.mean(np.array(episode_winners) == 2)

    print(
        f"Episode {episode + 1} finished | "
        f"Snake1={total1:.1f}, Snake2={total2:.1f}, Steps={steps} | "
        f"R1_ma10={avg_r1:.1f}, R2_ma10={avg_r2:.1f}, len_ma10={avg_len:.1f} | "
        f"win_rate1={win_rate1:.2f}, win_rate2={win_rate2:.2f}"
    )

plt.ioff()
plt.close(fig)
print("Simulation finished successfully.")