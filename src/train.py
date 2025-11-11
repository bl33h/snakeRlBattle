import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class SnakeEnv(gym.Env):
    def __init__(self, size=1000, max_food_distance=None):
        super(SnakeEnv, self).__init__()
        self.size = size
        self.max_food_distance = max_food_distance
        self.actionSpace = gym.spaces.Discrete(4)
        self.observationSpace = gym.spaces.Box(low=-1, high=1, shape=(24,), dtype=np.float32)
        self.reset()

    def reset(self):
        center = self.size // 2
        self.snake = [[center, center]]
        self.direction = 0
        self.done = False
        self.steps = 0
        self.score = 0
        
        # Generate obstacles - more structured patterns
        self.obstacles = self._generateObstacles()
        
        # Place food with curriculum support
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
        
        # Timeout penalty - increased base time
        max_steps = 500 + len(self.snake) * 50
        if self.steps > max_steps:
            self.done = True
            reward -= 5
        
        return self._getState(), reward, self.done, {}

    def _generateObstacles(self):
        """Generate structured obstacle patterns instead of random dots"""
        obstacles = set()
        
        # Reduce obstacles significantly for faster training
        num_patterns = max(2, self.size // 300)
        
        for _ in range(num_patterns):
            pattern_type = np.random.choice(['line', 'square'])  # Simplified patterns
            
            # Random starting position (avoid center where snake spawns)
            center = self.size // 2
            attempts = 0
            while attempts < 10:
                x = np.random.randint(self.size // 4, 3 * self.size // 4)
                y = np.random.randint(self.size // 4, 3 * self.size // 4)
                # Ensure not too close to center
                if abs(x - center) > 20 or abs(y - center) > 20:
                    break
                attempts += 1
            
            if pattern_type == 'line':
                # Horizontal or vertical line
                length = np.random.randint(5, 10)
                if np.random.random() < 0.5:  # Horizontal
                    for i in range(length):
                        if 0 <= x + i < self.size and 0 <= y < self.size:
                            obstacles.add((x + i, y))
                else:  # Vertical
                    for i in range(length):
                        if 0 <= x < self.size and 0 <= y + i < self.size:
                            obstacles.add((x, y + i))
            
            elif pattern_type == 'square':
                # Small square
                size = np.random.randint(3, 5)
                for i in range(size):
                    for j in range(size):
                        if 0 <= x + i < self.size and 0 <= y + j < self.size:
                            obstacles.add((x + i, y + j))
        
        # Remove obstacles that overlap with snake starting position
        center = self.size // 2
        obstacles = {obs for obs in obstacles if abs(obs[0] - center) > 5 or abs(obs[1] - center) > 5}
        
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
        state[21] = min(food_dist / 1000.0, 1.0)  # Distance to food
        state[22] = min(self.steps / 1000.0, 1.0)  # Steps since last food
        state[23] = self.score / 100.0  # Score (normalized)
        
        return state

    def _placeFood(self):
        """Place food with curriculum support - closer at first"""
        head = self.snake[0]
        
        if self.max_food_distance is None:
            # Random placement (late training)
            for _ in range(100):
                x = np.random.randint(0, self.size)
                y = np.random.randint(0, self.size)
                pos = [x, y]
                if pos not in self.snake and (x, y) not in self.obstacles:
                    return pos
        else:
            # Curriculum: place food within max_food_distance
            for _ in range(100):
                offset_x = np.random.randint(-self.max_food_distance, self.max_food_distance + 1)
                offset_y = np.random.randint(-self.max_food_distance, self.max_food_distance + 1)
                
                x = max(0, min(self.size - 1, head[0] + offset_x))
                y = max(0, min(self.size - 1, head[1] + offset_y))
                pos = [x, y]
                
                if pos not in self.snake and (x, y) not in self.obstacles:
                    return pos
        
        # Fallback
        return [(head[0] + 10) % self.size, (head[1] + 10) % self.size]

    def _isCollision(self, head):
        x, y = head
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return True
        # Only check recent body segments for large snakes (optimization)
        check_length = min(len(self.snake), 50)
        if head in self.snake[1:check_length]:
            return True
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
    total_episodes = 5000
    
    # Faster curriculum progression
    curriculum = [
        # (progress, grid_size, food_distance)
        (0.0, 50, 30),       # 0-10%: tiny grid, very close food
        (0.10, 100, 50),     # 10-25%: small grid, close food
        (0.25, 200, 100),    # 25-40%: medium grid, medium distance
        (0.40, 500, 200),    # 40-60%: large grid, far food
        (0.60, 1000, 500),   # 60-80%: huge grid, very far food
        (0.80, 1000, None),  # 80-100%: huge grid, random food
    ]
    
    state_dim = 24
    nActions = 4
    
    # GPU setup and verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"GPU CONFIGURATION")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"✓ Using device: {device}")
        # Enable cuDNN auto-tuner for optimal performance
        torch.backends.cudnn.benchmark = True
        print(f"✓ cuDNN Benchmark: Enabled")
    else:
        print(f"✗ CUDA not available, using CPU (training will be VERY slow)")
    print(f"{'='*60}\n")
    
    model = DQN(state_dim, nActions).to(device)
    target_model = DQN(state_dim, nActions).to(device)
    target_model.load_state_dict(model.state_dict())
    
    # Verify models are on GPU
    if torch.cuda.is_available():
        print(f"Model parameters on GPU: {next(model.parameters()).is_cuda}")
        print(f"Target model parameters on GPU: {next(target_model.parameters()).is_cuda}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    memory = deque(maxlen=100000)  # Increased buffer
    batchSize = 512  # INCREASED: 8GB VRAM can handle this easily
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    decay = 0.9995
    target_update = 10
    train_freq = 1
    
    # Mixed precision training for faster GPU computation
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print(f"✓ Mixed Precision Training (AMP): Enabled\n")
    
    best_score = 0
    episode_rewards = deque(maxlen=100)
    total_steps = 0  # Track total steps for efficient training
    
    # Start with smallest curriculum
    current_size = curriculum[0][1]
    current_food_dist = curriculum[0][2]
    env = SnakeEnv(size=current_size, max_food_distance=current_food_dist)
    
    print(f"Starting curriculum training over {total_episodes} episodes")
    print(f"Initial: {current_size}x{current_size} grid, food ±{current_food_dist} units\n")
    
    for episode in range(total_episodes):
        # Update curriculum based on progress
        progress = episode / total_episodes
        for threshold, size, food_dist in curriculum:
            if progress >= threshold:
                if size != current_size or food_dist != current_food_dist:
                    current_size = size
                    current_food_dist = food_dist
                    env = SnakeEnv(size=current_size, max_food_distance=current_food_dist)
                    percent = int(progress * 100)
                    food_str = "Random" if food_dist is None else f"±{food_dist}"
                    print(f"\n{'='*60}")
                    print(f"CURRICULUM UPDATE - {percent}% Complete")
                    print(f"Grid: {current_size}x{current_size} | Food: {food_str} units")
                    print(f"Episode: {episode} | Best Score: {best_score}")
                    print(f"{'='*60}\n")
        
        state = env.reset()
        totalReward = 0
        steps = 0
        
        while True:
            valid_actions = env.get_valid_actions()
            
            # Epsilon-greedy with valid actions only
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
            total_steps += 1
            
            # Training step - train more frequently once we have data
            if len(memory) >= batchSize and total_steps % train_freq == 0:
                batch = random.sample(memory, batchSize)
                states, actions, rewards, nextStates, dones = zip(*batch)
                
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                nextStates = torch.FloatTensor(np.array(nextStates)).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                # Use mixed precision for faster training
                if use_amp:
                    with torch.cuda.amp.autocast():
                        qValues = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                        
                        with torch.no_grad():
                            nextQ = target_model(nextStates).max(1)[0]
                            target = rewards + gamma * nextQ * (1 - dones)
                        
                        loss = nn.functional.mse_loss(qValues, target)
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    qValues = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    with torch.no_grad():
                        nextQ = target_model(nextStates).max(1)[0]
                        target = rewards + gamma * nextQ * (1 - dones)
                    
                    loss = nn.functional.mse_loss(qValues, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            
            if done:
                break
        
        epsilon = max(epsilon_min, epsilon * decay)
        episode_rewards.append(totalReward)
        
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
        if env.score > best_score:
            best_score = env.score
            torch.save(model.state_dict(), "snake_model_best.pth")
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            food_str = "Random" if current_food_dist is None else f"±{current_food_dist}"
            
            # Display GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"Ep {episode:4d} [{current_size:4d}x{current_size:4d}, Food: {food_str:>10}] | "
                      f"Reward: {totalReward:6.1f} | Avg: {avg_reward:6.1f} | "
                      f"Score: {env.score:2d} | Steps: {steps:4d} | "
                      f"Eps: {epsilon:.3f} | Best: {best_score:2d} | "
                      f"GPU: {memory_allocated:.2f}/{memory_reserved:.2f}GB")
            else:
                print(f"Ep {episode:4d} [{current_size:4d}x{current_size:4d}, Food: {food_str:>10}] | "
                      f"Reward: {totalReward:6.1f} | Avg: {avg_reward:6.1f} | "
                      f"Score: {env.score:2d} | Steps: {steps:4d} | "
                      f"Eps: {epsilon:.3f} | Best: {best_score:2d}")
    
    torch.save(model.state_dict(), "snake_model_final.pth")
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"Best score achieved: {best_score}")
    print(f"Final curriculum: {current_size}x{current_size} grid")
    print(f"Models saved: snake_model_best.pth, snake_model_final.pth")
    if torch.cuda.is_available():
        print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    print(f"{'='*60}")