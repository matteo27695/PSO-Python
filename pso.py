# ------------------------------------------------------------------------------+
#
#   Matteo Tortora
#   Particle Swarm Optimization (PSO) algorithm for functions optimization
#   February, 2020
#
# ------------------------------------------------------------------------------+

from random import random

import numpy as np


class Particle:
    def __init__(self, dim):
        self.pos = 100 * np.random.random(dim) - 50
        self.vel = np.zeros(dim)
        self.pbest_pos = self.pos
        self.pbest_val = np.inf

    def upd_pos(self):
        self.pos = self.pos + self.vel

    def upd_vel(self, w_inertia, w_cogn, w_soci, gbest_pos):
        inertia = w_inertia * self.vel
        bestind = w_soci * random() * (self.pbest_pos - self.pos)
        bestswa = w_cogn * random() * (gbest_pos - self.pos)
        self.vel = inertia + bestind + bestswa


class Swarm:
    def __init__(self, obj_fun, num_particles, n_iterations, dimspace, tolerance=0.0001, w_inertia=0.5, w_cogn=0.8, w_soci=0.9):
        self.w_inertia = w_inertia
        self.w_cogn = w_cogn
        self.w_soci = w_soci
        self.tolerance = tolerance
        self.n_iterations = n_iterations
        self.dimspace = dimspace
        self.gbest_pos = np.empty(self.dimspace)
        self.gbest_val = [np.inf]
        self.num_particles = num_particles
        self.particles = []
        self.obj_fun = obj_fun

    def init_particles(self):
        for i in range(self.num_particles):
            self.particles.append(Particle(self.dimspace))

    def fitness(self, particle):
        return self.obj_fun(particle.pos)

    def upd_pbest(self):
        for particle in self.particles:
            fitness_value = self.fitness(particle)
            if fitness_value < particle.pbest_val:
                particle.pbest_val = fitness_value
                particle.pbest_pos = particle.pos

    def upd_gbest(self):
        expected_gbest_val = self.gbest_val[-1]
        for particle in self.particles:
            if particle.pbest_val < expected_gbest_val:
                expected_gbest_val = particle.pbest_val
                expected_gbest_pos = particle.pbest_pos
        if expected_gbest_val < self.gbest_val[-1]:
            if self.gbest_val[-1] == np.inf:
                self.gbest_val[-1] = expected_gbest_val
                self.gbest_pos = expected_gbest_pos
            else:
                self.gbest_val.append(expected_gbest_val)
                self.gbest_pos = expected_gbest_pos

    def move_particles(self):
        for particle in self.particles:
            particle.upd_vel(self.w_inertia, self.w_cogn,
                             self.w_soci, self.gbest_pos)
            particle.upd_pos()

    def run(self):
        self.init_particles()
        i = 1
        while i < self.n_iterations + 1:
            self.upd_pbest()
            self.upd_gbest()
            self.move_particles()
            if i > 2 and len(self.gbest_val) > 2:
                if abs(self.gbest_val[-1] - self.gbest_val[-2]) < self.tolerance:
                    break
            i += 1
        print(
            f"The best position is: {self.gbest_pos} with value: {self.gbest_val[-1]}, in iteration number: {i}")


def example_fun(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    return x ** 2 + y ** 2 + z ** 2


if __name__ == "__main__":

    swarm = Swarm(obj_fun=example_fun, num_particles=20, dimspace=3, n_iterations=100)
    swarm.run()
