"""
Particle Swarm Optimization (PSO) - Basic Python Implementation
Solves a simple function minimization problem as a demo.
"""
import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

    def evaluate(self, func):
        value = func(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = np.copy(self.position)

    def update_velocity(self, global_best, w=0.5, c1=1, c2=2):
        r1, r2 = np.random.rand(), np.random.rand()
        self.velocity = (w * self.velocity +
                         c1 * r1 * (self.best_position - self.position) +
                         c2 * r2 * (global_best - self.position))

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

class PSO:
    def __init__(self, func, dim, bounds, num_particles=30, max_iter=100):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.swarm = [Particle(dim, bounds) for _ in range(num_particles)]
        self.global_best = np.random.uniform(bounds[0], bounds[1], dim)
        self.global_best_value = float('inf')

    def optimize(self):
        for i in range(self.max_iter):
            for particle in self.swarm:
                particle.evaluate(self.func)
                if particle.best_value < self.global_best_value:
                    self.global_best_value = particle.best_value
                    self.global_best = np.copy(particle.best_position)
            for particle in self.swarm:
                particle.update_velocity(self.global_best)
                particle.update_position(self.bounds)
        return self.global_best, self.global_best_value

# Example usage (minimize sphere function)
if __name__ == "__main__":
    def sphere(x):
        return sum(x**2)
    pso = PSO(sphere, dim=2, bounds=[-10, 10], num_particles=30, max_iter=100)
    best_pos, best_val = pso.optimize()
    print("Best position:", best_pos)
    print("Best value:", best_val)
