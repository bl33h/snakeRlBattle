import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class SnakeEnv(gym.Env):
    def __init__(self, size=1000):
        super(SnakeEnv, self).__init__()
        self.size = size
        self.actionSpace = gym.spaces.Discrete(4)
        # Use compact state representation instead of full grid
        self.observationSpace = gym.spaces.Box(low=-1, high=1, shape=(24,), dtype=np.float32)
        self.reset()

    def reset(self):
        center = self.size // 2
        self.snake = [[center, center]]
        self.direction = 0
        #self.food = self._placeFood()
        self.done = False
        self.steps = 0
        self.score = 0

        # aleatory obstacles
        self.obstacles = self._generateObstacles()

        # Place food avoiding obstacles and body
        self.food = self._placeFood()
        return self._getState()

    def step(self, action):
        if abs(action - self.direction) != 2:
            self.direction = action
        
        head = self.snake[0].copy()
        if self.direction == 0: head[1] -= 1
        elif self.direction == 1: head[0] += 1
        elif self.direction == 2: head[1] += 1
        else: head[0] -= 1
        
        reward = 0
        self.steps += 1
        
        # Check collision
        if self._isCollision(head):
            self.done = True
            reward = -10
            return self._getState(), reward, self.done, {}
        
        # Check if food eaten
        if head == self.food:
            self.snake.insert(0, head)
            self.score += 1
            reward = 10
            self.food = self._placeFood()
            self.steps = 0
        else:
            self.snake.insert(0, head)
            old_head = self.snake.pop()
            
            # Reward for moving toward food
            old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
            new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            if new_dist < old_dist:
                reward += 0.1
            else:
                reward -= 0.05
        
        # Timeout if snake is stuck (scaled for large grid and snake length)
        max_steps = 200 + len(self.snake) * 50
        if self.steps > max_steps:
            self.done = True
            reward -= 5
        
        return self._getState(), reward, self.done, {}

    def _generateObstacles(self):
        """
        Genera obstáculos aleatorios en el grid.
        La densidad se adapta al tamaño del entorno (menos obstáculos en entornos pequeños).
        """
        num_obstacles = max(5, self.size // 40)  # densidad proporcional
        obstacles = set()

        # Intentar colocar obstáculos en posiciones vacías
        while len(obstacles) < num_obstacles:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            pos = (x, y)
            if pos not in obstacles:
                if [x, y] not in self.snake:
                    obstacles.add(pos)

        return list(obstacles)

    def _getState(self):
        head = self.snake[0]
        state = np.zeros(24, dtype=np.float32)
        
        # 0-7: Danger in 8 directions (binary)
        directions = [
            [0, -1], [1, -1], [1, 0], [1, 1],
            [0, 1], [-1, 1], [-1, 0], [-1, -1]
        ]
        for i, (dx, dy) in enumerate(directions):
            check_pos = [head[0] + dx, head[1] + dy]
            state[i] = 1.0 if self._isCollision(check_pos) else 0.0
        
        # 8-11: Current direction (one-hot)
        state[8 + self.direction] = 1.0
        
        # 12-15: Food direction (normalized)
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        food_dist = max(abs(food_dx), abs(food_dy), 1)
        
        state[12] = 1.0 if food_dy < 0 else 0.0  # Food up
        state[13] = 1.0 if food_dx > 0 else 0.0  # Food right
        state[14] = 1.0 if food_dy > 0 else 0.0  # Food down
        state[15] = 1.0 if food_dx < 0 else 0.0  # Food left
        
        # 16-19: Distance to walls (normalized)
        state[16] = head[1] / self.size  # Distance to top
        state[17] = (self.size - head[0] - 1) / self.size  # Distance to right
        state[18] = (self.size - head[1] - 1) / self.size  # Distance to bottom
        state[19] = head[0] / self.size  # Distance to left
        
        # 20-23: Additional features
        state[20] = min(len(self.snake) / 100.0, 1.0)  # Snake length (capped)
        state[21] = min(food_dist / 1000.0, 1.0)  # Distance to food (normalized for large grid)
        state[22] = min(self.steps / 1000.0, 1.0)  # Steps since last food
        state[23] = self.score / 100.0  # Score (normalized)
        
        return state

    def _placeFood(self):
        """Optimized food placement - start close for early training"""
        head = self.snake[0]
        for _ in range(100):
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            pos = [x, y]
            if pos not in self.snake and (x, y) not in self.obstacles:
                return pos
        return [(head[0] + 10) % self.size, (head[1] + 10) % self.size]

    def _isCollision(self, head):
        x, y = head
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return True
        if head in self.snake[1:min(len(self.snake), 100)]:
            return True
        # collision with obstacles
        if (x, y) in self.obstacles:
            return True
        return False

    def get_valid_actions(self):
        valid = []
        head = self.snake[0].copy()
        for action in range(4):
            if abs(action - self.direction) == 2:
                continue
            test_head = head.copy()
            if action == 0: test_head[1] -= 1
            elif action == 1: test_head[0] += 1
            elif action == 2: test_head[1] += 1
            else: test_head[0] -= 1
            if not self._isCollision(test_head):
                valid.append(action)
        return valid if valid else [self.direction]


class DQN(nn.Module):
    def __init__(self, state_dim, nActions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, nActions)
        )
    
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    total_episodes = 6000
    
    # Curriculum learning: start small, scale up percentually
    curriculum = [
        (0.0, 50),      # 0-10%: 50x50
        (0.10, 100),    # 10-25%: 100x100
        (0.25, 200),    # 25-40%: 200x200
        (0.40, 500),    # 40-65%: 500x500
        (0.65, 1000),   # 65-100%: 1000x1000
    ]
    
    state_dim = 24
    nActions = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DQN(state_dim, nActions).to(device)
    target_model = DQN(state_dim, nActions).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    memory = deque(maxlen=50000)
    batchSize = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    decay = 0.9995
    target_update = 20
    train_freq = 4  # Train every N steps instead of every step
    
    best_score = 0
    episode_rewards = deque(maxlen=100)
    
    # Start with smallest grid
    current_size = curriculum[0][1]
    env = SnakeEnv(size=current_size)
    print(f"Starting curriculum training over {total_episodes} episodes")
    print(f"Initial grid: {current_size}x{current_size}")
    
    for episode in range(total_episodes):
        # Update grid size based on curriculum percentage
        progress = episode / total_episodes
        for threshold, size in curriculum:
            if progress >= threshold:
                if size != current_size:
                    current_size = size
                    env = SnakeEnv(size=current_size)
                    percent = int(progress * 100)
                    print(f"\n=== {percent}% - Scaling up to {current_size}x{current_size} grid (episode {episode}) ===\n")
        state = env.reset()
        totalReward = 0
        steps = 0
        step_count = 0
        
        while True:
            valid_actions = env.get_valid_actions()
            
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q = model(state_tensor)
                    q_masked = q[0].clone()
                    invalid = [a for a in range(nActions) if a not in valid_actions]
                    q_masked[invalid] = -float('inf')
                    action = torch.argmax(q_masked).item()
            
            nextState, reward, done, _ = env.step(action)
            memory.append((state, action, reward, nextState, done))
            state = nextState
            totalReward += reward
            steps += 1
            step_count += 1
            
            if done:
                break
            
            # Training step (only every train_freq steps for speed)
            if len(memory) > batchSize * 2 and step_count % train_freq == 0:
                batch = random.sample(memory, batchSize)
                states, actions, rewards, nextStates, dones = zip(*batch)
                
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                nextStates = torch.FloatTensor(np.array(nextStates)).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                qValues = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    nextQ = target_model(nextStates).max(1)[0]
                    target = rewards + gamma * nextQ * (1 - dones)
                
                loss = nn.functional.mse_loss(qValues, target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        epsilon = max(epsilon_min, epsilon * decay)
        episode_rewards.append(totalReward)
        
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
        if env.score > best_score:
            best_score = env.score
            torch.save(model.state_dict(), "snake_model_best.pth")
        
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"Ep {episode} [{current_size}x{current_size}]: reward={totalReward:.1f}, "
                  f"avg={avg_reward:.1f}, score={env.score}, steps={steps}, "
                  f"eps={epsilon:.3f}, best={best_score}")
    
    torch.save(model.state_dict(), "snake_model_final.pth")
    print(f"\nTraining complete! Best score: {best_score}")
    print(f"Final grid size: {current_size}x{current_size}")
    print("Models saved as snake_model_best.pth and snake_model_final.pth")