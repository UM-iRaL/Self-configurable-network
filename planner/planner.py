import numpy as np
import time
import networkx as nx

# from main import N_AGENTS

class Planner(object):

    def __init__(self):
        pass

    def compute_cost(self, agents):
        """
        Computes the cost of a given agent configuration.
        :param agents: The set of agents
        :return: The coverage cost function evaluated at the current agent states
        """
        observed_points = set()  # Use set unions of the observed points to compute the objective.
        for agent in agents:
            observed_points = observed_points.union(agent.get_observations(agent.state))
        return len(observed_points)


    def connectivity_graph(self, agents, comm_range):
        """
        Construct connectivity graph for agents.
        :param agents: The set of agents
        :param comm_range: Communication range of agents
        :return: Connectivity graph
        """
        G = nx.Graph()
        for idx_i, i in enumerate(agents):
            G.add_node(idx_i)
        for idx_i, i in enumerate(agents):
            for idx_j, j in enumerate(agents):
                if np.linalg.norm((i.state[0] - j.state[0], i.state[1] - j.state[1])) < comm_range \
                and i != j:
                    G.add_edge(idx_i,idx_j)
        return G


    def plan_rag(self, agents, comm_range):
        """
        Performs Resource-Aware distributed Greedy (RAG) for each agent.
        :param agents: The set of agents
        :param comm_range: Communication range of agents
        :return: Number of communication rounds and number of computations
        """
        observed_points = set()
        agents_selected = set()
        n_round = 0

        # Reset connectivity
        for i in agents:
            i.in_neighbors = set()
            for j in agents:
                if np.linalg.norm((i.state[0] - j.state[0], i.state[1] - j.state[1])) < comm_range \
                and i != j:
                    i.in_neighbors.add(j)
        
        while not len(agents_selected) == len(agents):   
            # print('already selected : ', len(agents_selected))         
            # Update locally best actions and gains
            self.update_action_gain(agents, observed_points)
            
            # Agents select for one iteration
            observed_points = self.selection_iteration(agents, observed_points, agents_selected)

            n_round += 2

        n_round = n_round - 2 # no communication round in the last iteration

        return len(observed_points), n_round


    def selection_iteration(self, agents, observed_points, agents_selected):
        """
        Performs one selection iteration for unselected agents
        :param agents: Set of agents to plan for
        :param observed_points: Set of currently observed points
        :param agents_selected: Set of agents that already selected actions
        :return: Set of observed points after this selection iteration
        """
        for i in agents:
            i.unchecked  = 1 # reset agents checked in the last iteration 

        for i in agents:
            if i.unselected  == 0 or i.unchecked  == 0: # only unselected and 
                continue                                # unchecked agents select 

            i.unselected  = 0 
            for j in i.in_neighbors:
                if j.unselected == 1: # compare with unselected in-neighbors only
                    if i.gain < j.gain:
                        i.unselected  = 1 # i isn't local maximum so doesn't select
                        break
            # if i selects
            if i.unselected  == 0:
                i.unchecked = 0
                for j in i.in_neighbors:
                    j.unchecked = 0
                observed_points = observed_points.union(i.next_observed_points)    
                agents_selected.add(i)
        return observed_points


    def update_action_gain(self, agents, observed_points):
        """
        Updates locally best actions and gains for unselected agents
        :param agents: Set of agents to plan for.
        :param observed_points : Set of currently observed points
        :return: None
        """
        for i in agents:
            if i.unselected == 0: # update for unselected agents only
                continue
            best_action = []
            best_gain = -1
            best_observed_points = set()
            for succ, action in i.get_successors():
                succ_best_gain = len(observed_points.union(i.get_observations(succ)))
                if succ_best_gain > best_gain:
                    best_action = action
                    best_gain = succ_best_gain
                    best_observed_points = i.get_observations(succ)
            i.next_observed_points = best_observed_points
            observed_points = observed_points.union(best_observed_points)
            i.set_next_action(best_action)
            i.gain = best_gain


    def plan_sga(self, agents):
        """
        Performs Sequential Greedy Assignment as a baseline algorithm.
        :param agents: Set of agents to plan for.
        :param radius:
        :return: None
        """
        observed_points = set()
        for agent in agents:
            best_action = []
            best_cost = -1
            best_set = set()
            for succ, action in agent.get_successors():
                succ_best_cost = len(observed_points.union(agent.get_observations(succ)))
                if succ_best_cost > best_cost:
                    best_action = action
                    best_cost = succ_best_cost
                    best_set = agent.get_observations(succ)
            observed_points = observed_points.union(best_set)
            agent.set_next_action(best_action)


    def check_cost(self, x, agents):
        observed_points = set()
        for num, agent in enumerate(agents):
            actions_prob = x[num * agent.n_actions: num * agent.n_actions + agent.n_actions]
            for idx, succ_act in enumerate(agent.get_successors()):
                if actions_prob[idx] == 1:
                    observed_points = observed_points.union(agent.get_observations(succ_act[0]))
        return len(observed_points)



# if __name__ == "__main__":
#     planner = Planner()
#     v = 1.0 * np.array([1000, 500, 50, 100, 100])
#     v = np.hstack((v, v))
#     print('Vector', v, 'Projection=', planner.project_P(v, 1, 5))
