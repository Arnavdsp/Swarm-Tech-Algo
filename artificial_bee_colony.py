"""
Artificial Bee Colony (ABC) Algorithm - Basic Python Implementation
Solves a simple function minimization problem as a demo.
"""
import numpy as np

class FoodSource:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.fitness = float('inf')
        self.trial = 0

    def evaluate(self, func):
        self.fitness = func(self.position)

class ABC:
    def __init__(self, func, dim, bounds, num_bees=20, max_iter=100):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.num_bees = num_bees
        self.max_iter = max_iter
        self.food_sources = [FoodSource(dim, bounds) for _ in range(num_bees)]
        self.best_source = None

    def optimize(self):
        for fs in self.food_sources:
            fs.evaluate(self.func)
        for it in range(self.max_iter):
            # Employed bees
            for fs in self.food_sources:
                k = np.random.randint(self.num_bees)
                phi = np.random.uniform(-1, 1, self.dim)
                new_pos = fs.position + phi * (fs.position - self.food_sources[k].position)
                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                new_fit = self.func(new_pos)
                if new_fit < fs.fitness:
                    fs.position = new_pos
                    fs.fitness = new_fit
                    fs.trial = 0
                else:
                    fs.trial += 1
            # Onlooker bees
            fitnesses = np.array([1/(fs.fitness+1e-6) for fs in self.food_sources])
            probs = fitnesses / fitnesses.sum()
            for i in range(self.num_bees):
                fs = np.random.choice(self.food_sources, p=probs)
                k = np.random.randint(self.num_bees)
                phi = np.random.uniform(-1, 1, self.dim)
                new_pos = fs.position + phi * (fs.position - self.food_sources[k].position)
                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                new_fit = self.func(new_pos)
                if new_fit < fs.fitness:
                    fs.position = new_pos
                    fs.fitness = new_fit
                    fs.trial = 0
                else:
                    fs.trial += 1
            # Scout bees
            for fs in self.food_sources:
                if fs.trial > 10:
                    fs.position = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                    fs.fitness = self.func(fs.position)
                    fs.trial = 0
        self.best_source = min(self.food_sources, key=lambda fs: fs.fitness)
        return self.best_source.position, self.best_source.fitness

# Example usage (minimize sphere function)
if __name__ == "__main__":
    def sphere(x):
        return sum(x**2)
    abc = ABC(sphere, dim=2, bounds=[-10, 10], num_bees=20, max_iter=100)
    best_pos, best_val = abc.optimize()
    print("Best position:", best_pos)
    print("Best value:", best_val)
