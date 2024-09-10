import matplotlib.pyplot as plt
import matplotlib.patches as patch
import agent.agent as agent

class Simulator(object):

    def __init__(self, agents, height, width):
        self.agents = agents
        self.height = height
        self.width = width

        # Plotting
        self.fig = plt.figure(1)
        self.ax = self.fig.gca()

        plt.ion()

    def simulate(self, T=1):
        """
        Simulate T timesteps.
        :param T: Number of timesteps to simulate
        :return: None
        """
        for agent in self.agents:
            agent.apply_next_action()


    def compute_cost(self):
        pass

    def draw(self):
        for agent in self.agents:
            self.ax.add_patch(agent.patch)
            self.ax.scatter(x=agent.state[0], y=agent.state[1], marker='*', color='k')
            [self.ax.add_patch(obs_point) for obs_point in agent.obs_patches]

        self.ax.set_xlim([0, self.width])
        self.ax.set_ylim([0, self.height])
        plt.xticks(fontname="Times New Roman", fontsize=18)
        plt.yticks(fontname="Times New Roman", fontsize=18)

        # self.ax.set_title('Image Covering')
        # self.ax.set_xlabel('X')
        # self.ax.set_ylabel('Y')
        self.ax.set_aspect('equal')
