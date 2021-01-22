import matplotlib.pyplot as plt

import pso


def example_fun(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    return x ** 2 + y ** 2 + z ** 2


swarm = pso.Swarm(obj_fun=example_fun, num_particles=20, dimspace=3, n_iterations=100)
swarm.run()
plt.plot(swarm.gbest_val)
plt.xlabel('Number of Iterations')
plt.ylabel('Global Best Value')
plt.grid(True)
plt.show()
