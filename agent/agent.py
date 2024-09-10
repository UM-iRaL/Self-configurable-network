import numpy as np
import math
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from agent.observation_model import ObservationModel


class Agent(object):

    def __init__(self, state, radius, comm_range, height, width, res, n_time_step, color):
        self.state = np.array(state)        # [x_1; x_2]
        self.radius = radius      # limited sensing radius
        self.comm_range = comm_range

        # self.time_step = time_step
        # print(self.time_step)

        self.traj = [np.array(state)]
        self.action_hist = []

        # Action Space
        # self.actions = {'stay': (0, 0), 'up': (0, 2), 'down': (0, -3),
        #                 'left': (-1, 0), 'right': (4, 0)}
        # self.n_actions = 6 # len(self.actions)
        # step = np.random.uniform(1.0, 5.0)
        # directions = np.random.uniform(0.0, math.pi, self.n_actions)
        directions = np.linspace(0.0, 1.75*math.pi, 8)
        step = radius
        # print(directions)
        # [(step * math.cos(theta), step * math.sin(theta)) for theta in directions] #
        # self.actions = [(0, 0), (0, 9), (0, -5), (3, -3), (-3, -5), (7, 0)]
        # self.actions = step * np.array([np.cos(directions), np.sin(directions)]).T
        self.actions = [(step * np.cos(theta), step * np.sin(theta)) for theta in directions]
        # print(self.actions)
        self.n_actions = len(self.actions)
        self.action_indices = np.array(range(self.n_actions))
        self.next_action_index = 0
        self.next_placement = self.state

        # Observation Model
        self.observation_model = ObservationModel(int(radius), height, width, res)

        # Neighborhood
        self.in_neighbor_candidates = []
        self.in_neighborhood_candidate_size = 0
        self.in_neighbors = [[] for i in range(n_time_step)]
        self.in_neighborhood_size = 0
        # self.next_neighbor_indices = set()
        self.neighbor_select_flag = True

        # Plotting
        self.color = color

        # OSG
        self.eta_act = np.sqrt(8 * math.log(self.n_actions) / n_time_step) # learning rate for selecting actions
        self.action_weight = [[1.0 / self.n_actions for i in range(self.n_actions)] for j in range(n_time_step+1)]  # weights of actions       
        # self.action_prob_dist = [[0.0 for i in range(self.n_actions)] for j in range(n_time_step)] 
        self.action_loss = [[0.0 for i in range(self.n_actions)] for j in range(n_time_step)] 
                
        self.eta_nei = 0 # np.sqrt(2 * math.log(self.in_neighborhood_candidate_size) / self.in_neighborhood_candidate_size / n_time_step) # learning rate for selecting neighbors
        self.gamma_nei = self.eta_nei / 2 # bias for selecting neighbors
        self.neighbor_weight = [[[1.0 for i in range(self.in_neighborhood_candidate_size)] for j in range(self.in_neighborhood_size)] for t in range(n_time_step+1)]  # weights of neighbor candidates for all EXP3-IX
        self.neighbor_loss = [[[0.0 for i in range(self.in_neighborhood_candidate_size)] for j in range(self.in_neighborhood_size)] for t in range(n_time_step+1)]  # weights of neighbor candidates for all EXP3-IX


    def get_action_losses(self, t):
        """
        Returns the losses of all possible actions based on the estimation result of just executed actions
        :return: The losses of all possible actions.
        """
        # losses = np.zeros(self.n_actions)
        obj_all_actions = np.zeros(self.n_actions)
        next_placements = [self.motion_model(self.state, self.actions[idx]) for idx in range(self.n_actions)]
        # print('state: ', self.state)
        # print('next_placements: ', next_placements)

        # if len(self.in_neighbors[t]) == 0:
        if self.neighbor_select_flag == False:
            # print('No neighbors')
            obj_all_actions = np.array([len(self.get_observations(placement)) for placement in next_placements])
        else:
            # print('number of neighbors: ', len(self.in_neighbors))
            neighbors_observed_points = set()
            for neighbor in self.in_neighbors[t]:
                neighbors_observed_points = neighbors_observed_points.union(neighbor.get_observations(neighbor.next_placement))
            
            for idx in range(self.n_actions):
                obj_all_actions[idx] = len(neighbors_observed_points.union(self.get_observations(next_placements[idx]))) - len(neighbors_observed_points)
                
        # print(obj_all_actions)
        if max(obj_all_actions) == 0:
            # no action has any reward (marginal gain)
            losses = np.zeros(self.n_actions)
        else:
            losses = -np.array(obj_all_actions) / max(obj_all_actions)
        self.action_loss[t] = losses 

    # def get_action_prob_dist(self, t):
    #     """
    #     Returns the output of FSF* (the predicted action probability distribution)
    #     :param t: The index of time step.
    #     :return: None.
    #     """
    #     self.action_prob_dist[t] = self.action_weight[t] # m x 1

    def apply_next_action(self):
        """
        Applies the next action to modify the agent state.
        :param t: Time step.
        :return: None
        """
        self.state = self.motion_model(self.state, self.actions[self.next_action_index])

    def update_action_weights(self, t):
        """
        Updates the parameters of experts after getting losses (from t to t + 1)
        :param t: The index of time step
        :return: None
        """
        self.action_weight[t + 1] = [self.action_weight[t][i] * np.exp(-self.eta_act * self.action_loss[t][i]) for i in range(self.n_actions)]

        self.action_weight[t + 1] = self.action_weight[t + 1] / np.linalg.norm(self.action_weight[t + 1], ord=1)

    def get_neighbor_losses(self, t, idx, neighbors_selected, next_neighbor_index):
        """
        Returns the losses of all possible actions based on the estimation result of just executed actions
        :return: The losses of all possible actions.
        """
        next_neighbor = self.in_neighbor_candidates[next_neighbor_index]
        # if len(neighbors_selected) == 0:
        #     r = len(self.get_observations(self.next_placement)) + len(self.get_observations(next_neighbor.next_placement)) - len(self.get_observations(self.next_placement).union(self.get_observations(next_neighbor.next_placement)))
        # else:
        # print('number of neighbors: ', len(self.in_neighbors))
        neighbors_selected_observed_points = set()
        for neighbor in neighbors_selected:
            neighbors_selected_observed_points = neighbors_selected_observed_points.union(neighbor.get_observations(neighbor.next_placement))

        new_neighborhood_observed_points = neighbors_selected_observed_points.union(self.get_observations(next_neighbor.next_placement))

        r = -len(new_neighborhood_observed_points.union(self.next_placement)) + len(new_neighborhood_observed_points) + len(neighbors_selected_observed_points.union(self.next_placement)) - len(neighbors_selected_observed_points)
        
        r_max = len(self.get_observations(self.next_placement)) + len(self.get_observations(next_neighbor.next_placement)) - len(self.get_observations(self.next_placement).union(self.get_observations(next_neighbor.next_placement)))

        if r_max != 0:
            r = r / r_max
        l = 1 - r
        l_tilda = l / (self.neighbor_weight[t][idx][next_neighbor_index] + self.gamma_nei)

        self.neighbor_loss[t][idx][next_neighbor_index] = l_tilda 

    def update_neighbor_weights(self, t, idx):
        """
        Updates the parameters of experts after getting losses (from t to t + 1)
        :param t: The index of time step
        :return: None
        """

        self.neighbor_weight[t + 1][idx] = [self.neighbor_weight[t][idx][i] * np.exp(self.eta_nei * (1 - self.neighbor_loss[t][idx][i])) for i in range(self.in_neighborhood_candidate_size)]

        self.neighbor_weight[t + 1][idx] = self.neighbor_weight[t + 1][idx] / np.linalg.norm(self.neighbor_weight[t + 1][idx], ord=1)

    def get_successors(self):
        """
        Returns possible subsequent states along each valid action, given the current state.
        :return: The list of subequent states.
        """
        return [self.motion_model(self.state, action) for action in self.actions]

    def get_observations(self, state):
        """
        Returns the observations at a potential new state.
        :param state: The state to observe from.
        :return: The set of observed points at the new state
        """
        return self.observation_model.get_observed_points(state)

    def set_next_action(self, action):
        """
        Assign next action.
        :param action: The action to assign
        :return: None
        """
        self.next_action = action

    def get_next_placement(self):
        """
        Applies the next action to modify the agent state.
        :return: None
        """
        self.next_placement = self.motion_model(self.state, self.actions[self.next_action])

    def motion_model(self, state, action):
        '''
        :param state: The current state at time t.
        :param action: The current action at time t.
        :return: The resulting state x_{t+1}
        '''
        return state[0] + action[0], state[1] + action[1]