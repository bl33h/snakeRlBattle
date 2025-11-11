import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from train import SnakeEnv, DQN

class SnakeEnvEasy(SnakeEnv):
    """Extended environment with curriculum-based food placement for visualization"""
    
    def __init__(self, size=1000, max_food_distance=None):
        self.max_food_distance = max_food_distance
        super().__init__(size=size)
    
    def _placeFood(self):
        """Place food within a certain distance from the snake head"""
        head = self.snake[0]
        
        if self.max_food_distance is None:
            # Default behavior - completely random
            return super()._placeFood()
        
        # Try to place food within max_food_distance
        for _ in range(100):
            # Random position within a box around the head
            offset_x = np.random.randint(-self.max_food_distance, self.max_food_distance + 1)
            offset_y = np.random.randint(-self.max_food_distance, self.max_food_distance + 1)
            
            x = max(0, min(self.size - 1, head[0] + offset_x))
            y = max(0, min(self.size - 1, head[1] + offset_y))
            pos = [x, y]
            
            if pos not in self.snake and (x, y) not in self.obstacles:
                return pos
        
        # Fallback to nearby position
        return [(head[0] + 10) % self.size, (head[1] + 10) % self.size]
    
    def update_difficulty(self, score):
        """Update max_food_distance based on current score"""
        if score < 3:
            self.max_food_distance = 50
        elif score < 6:
            self.max_food_distance = 100
        elif score < 10:
            self.max_food_distance = 200
        elif score < 15:
            self.max_food_distance = 500
        else:
            self.max_food_distance = None  # Full random

state_dim = 24
nActions = 4

model = DQN(state_dim, nActions)
model.load_state_dict(torch.load("snake_model_best.pth"))
model.eval()

# Create environment starting with easiest difficulty
env = SnakeEnvEasy(size=1000, max_food_distance=50)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))

state = env.reset()
totalReward = 0
step_num = 0
previous_score = 0

print("🎮 Starting adaptive difficulty snake game...")
print("Difficulty increases as you eat more food!\n")

while True:
    ax.clear()
    
    grid_size = env.size
    head = env.snake[0]
    
    # Focus camera on snake with some padding
    view_size = 150  # Show 150x150 area around snake
    x_min = max(0, head[0] - view_size)
    x_max = min(grid_size, head[0] + view_size)
    y_min = max(0, head[1] - view_size)
    y_max = min(grid_size, head[1] + view_size)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    # Draw obstacles (only visible ones)
    for obs in env.obstacles:
        if x_min <= obs[0] <= x_max and y_min <= obs[1] <= y_max:
            rect = Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, 
                           facecolor='gray', edgecolor='black')
            ax.add_patch(rect)
    
    # Draw snake body
    for segment in env.snake[1:]:
        rect = Rectangle((segment[0] - 0.5, segment[1] - 0.5), 1, 1,
                       facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
        ax.add_patch(rect)
    
    # Draw snake head
    rect = Rectangle((head[0] - 0.5, head[1] - 0.5), 1, 1,
                   facecolor='darkgreen', edgecolor='black', linewidth=3)
    ax.add_patch(rect)
    
    # Draw food
    food = env.food
    rect = Rectangle((food[0] - 0.5, food[1] - 0.5), 1, 1,
                   facecolor='red', edgecolor='darkred', linewidth=2)
    ax.add_patch(rect)
    
    # Calculate distance to food
    food_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
    
    # Calculate steps until timeout
    max_steps = 200 + len(env.snake) * 50
    steps_remaining = max_steps - env.steps
    
    # Get difficulty level description
    if env.max_food_distance is None:
        difficulty = "MASTER (Full Map)"
        diff_color = "red"
    elif env.max_food_distance >= 500:
        difficulty = f"HARD (±{env.max_food_distance})"
        diff_color = "orange"
    elif env.max_food_distance >= 200:
        difficulty = f"MEDIUM (±{env.max_food_distance})"
        diff_color = "yellow"
    elif env.max_food_distance >= 100:
        difficulty = f"EASY (±{env.max_food_distance})"
        diff_color = "lightgreen"
    else:
        difficulty = f"BEGINNER (±{env.max_food_distance})"
        diff_color = "lightblue"
    
    plt.title(f"Adaptive Snake AI | Difficulty: {difficulty}\n"
             f"Step {step_num} | Score: {env.score} | Length: {len(env.snake)} | "
             f"Reward: {totalReward:.1f}\nFood Distance: {food_dist} | "
             f"Timeout in: {steps_remaining}", 
             fontsize=10)
    
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    plt.pause(0.05)
    
    # Get action from model
    with torch.no_grad():
        state_flat = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
        q = model(state_flat)
        action = torch.argmax(q).item()
    
    # Step environment
    nextState, reward, done, _ = env.step(action)
    
    # Check if score increased (food was eaten)
    if env.score > previous_score:
        previous_score = env.score
        old_difficulty = env.max_food_distance
        env.update_difficulty(env.score)
        if env.max_food_distance != old_difficulty:
            new_diff = "Full Random" if env.max_food_distance is None else f"±{env.max_food_distance}"
            print(f"🎯 Score: {env.score} | Difficulty increased to {new_diff}!")
    
    state = nextState
    totalReward += reward
    step_num += 1
    
    if done:
        # Print why it died
        x, y = head
        
        print("\n" + "="*50)
        if x < 0 or y < 0 or x >= env.size or y >= env.size:
            print(f"Game Over: Hit wall at {head}")
        elif head in env.snake[1:]:
            print(f"Game Over: Hit own body at {head}")
        elif (x, y) in env.obstacles:
            print(f"Game Over: Hit obstacle at {head}")
        elif env.steps >= max_steps:
            print(f"Game Over: TIMEOUT - {env.steps} steps without eating")
        else:
            print(f"Game Over: Unknown reason")
        
        print("="*50)
        print(f"Final Stats:")
        print(f"Score: {env.score}")
        print(f"Length: {len(env.snake)}")
        print(f"Total Reward: {totalReward:.1f}")
        print(f"Steps Taken: {step_num}")
        print(f"Final Difficulty: {difficulty}")
        print("="*50)
        
        plt.pause(2.0)
        break

plt.ioff()
plt.show()

print("\n🎮 Playback complete!")