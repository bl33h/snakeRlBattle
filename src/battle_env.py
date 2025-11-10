import numpy as np
from train import SnakeEnv

class SnakeBattleEnv:
    def __init__(self, size=400):
        # larger arena for visibility
        self.size = size
        self.env1 = SnakeEnv(size)
        self.env2 = SnakeEnv(size)
        self.done = False
        self.food = self.env1.food

    def reset(self):
        # reset both and place snakes far apart, random directions
        self.env1.reset()
        self.env2.reset()

        margin = self.size // 5
        self.env1.snake = [[np.random.randint(margin, self.size // 2),
                             np.random.randint(margin, self.size // 2)]]
        self.env2.snake = [[np.random.randint(self.size // 2, self.size - margin),
                             np.random.randint(self.size // 2, self.size - margin)]]

        self.env1.direction = np.random.randint(0, 4)
        self.env2.direction = np.random.randint(0, 4)

        self.food = self.env1._placeFood()
        self.done = False
        return self.env1._getState(), self.env2._getState()

    def step(self, a1, a2):
        # share same food
        self.env1.food = self.food
        self.env2.food = self.food

        s1, r1, d1, _ = self.env1.step(a1)
        s2, r2, d2, _ = self.env2.step(a2)

        # competition logic
        if self.env1.snake[0] == self.food and self.env2.snake[0] != self.food:
            r1 += 5
            r2 -= 2
            self.food = self.env1.food
        elif self.env2.snake[0] == self.food and self.env1.snake[0] != self.food:
            r2 += 5
            r1 -= 2
            self.food = self.env2.food

        # collisions with each other
        if self.env1.snake[0] in self.env2.snake:
            r1 -= 10
            d1 = True
        if self.env2.snake[0] in self.env1.snake:
            r2 -= 10
            d2 = True

        self.done = d1 or d2
        return s1, s2, r1, r2, self.done

    def renderBoard(self):
        scale = max(2, 800 // self.size)  # scale for clear visuals
        board = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # draw thicker green snake
        for x, y in self.env1.snake:
            if 0 <= x < self.size and 0 <= y < self.size:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < self.size and 0 <= yy < self.size:
                            board[yy, xx] = [0, 255, 100]

        # draw thicker blue snake
        for x, y in self.env2.snake:
            if 0 <= x < self.size and 0 <= y < self.size:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < self.size and 0 <= yy < self.size:
                            board[yy, xx] = [0, 100, 255]

        # draw larger red food
        fx, fy = self.food
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                xx, yy = fx + dx, fy + dy
                if 0 <= xx < self.size and 0 <= yy < self.size:
                    board[yy, xx] = [255, 50, 50]

        # upscale for visibility
        if scale > 1:
            board = board.repeat(scale, axis=0).repeat(scale, axis=1)

        return board