import matplotlib.pyplot as plt

import pso_ode

swarm = pso_ode.Swarm(num_particles=20, dimspace=2, n_iterations=100)
swarm.run()
plt.plot(swarm.gbest_val)
plt.xlabel('Number of Iterations')
plt.ylabel('Global Best Value')
plt.grid(True)
plt.show()
