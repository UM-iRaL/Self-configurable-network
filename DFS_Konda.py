from sympy import ordered
import simulator
from agent.agent import Agent
from simulator.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
from planner.planner import Planner
import time
import psutil
from memory_profiler import memory_usage
import networkx as nx

SIM_STEPS = 1
N_AGENTS = 100
WIDTH = 300
HEIGHT = 30
RADIUS = 15
COMM_RANGE = 20
STEPSIZE = 1
RES = 1

def main():
    # for rnd_seed in range(1, 2):
    #     np.random.seed(rnd_seed)
    #     coverage, round = trial()
    #     print('Coverage is :', coverage)
    #     print('Comm Round is :', round)
    #     # plt.plot(range(SIM_STEPS+1), coverage)

    # plt.show()
    # plt.pause(100)
    '''
    random connected graph
    '''
    all_n_round = []    
    all_n_coverage = []
    all_time = []
    all_memory = []
    for rnd_seed in range(1, 51):
        np.random.seed(rnd_seed)
        # coverage, round = trial()
        # agents = [create_agent() for i in range(0, N_AGENTS)]
        # agents = create_sparse_agent()
        # graph = connectivity_graph(agents, comm_range=COMM_RANGE)
        agents, graph = connected_connectivity_graph()
        planner = Planner()

        start_time = time.time()
        # n_round, ordered_agents = dfs_order(graph, agents)
        # n_coverage = plan_sga(ordered_agents)

        n_round, n_coverage = plan_dfs_sga(graph, agents)
        # for testing correctness of coverage
        # sim = Simulator(agents=agents, height=HEIGHT, width=WIDTH)
        # sim.simulate()
        # print(planner.compute_cost(agents))

        all_memory.append(np.round(psutil.Process().memory_info().rss / 1024 ** 2, 2))

        all_time.append(time.time() - start_time)
        all_n_round.append(n_round)
        all_n_coverage.append(n_coverage)
        # print('Coverage is :', coverage)
        # print('Comm Round is :', round)
        # plt.plot(range(SIM_STEPS+1), coverage)
        
    print('Aver Comm Round is :', np.mean(all_n_round)) 
    print('Aver Coverage is :', np.mean(all_n_coverage)) 
    print('Aver Time is :', np.mean(all_time)) 
    print('Aver Memory is :', np.mean(all_memory))

    # # plt.boxplot(np.array(all_round))
    # plt.hist(np.array(all_round))
    # plt.show()

def sparse_agents_RAG_DFS_compare():
    '''
    '''
    all_n_round = []    
    all_n_coverage = []
    all_time = []
    all_memory = []
    for rnd_seed in range(1, 51):
        # np.random.seed(rnd_seed)
        # coverage, round = trial()
        # agents = [create_agent() for i in range(0, N_AGENTS)]
        # graph = connectivity_graph(agents, comm_range=COMM_RANGE)
        agents = create_sparse_agent()
        graph = connectivity_graph(agents, comm_range=25)
        planner = Planner()

        start_time = time.time()
        # n_round, ordered_agents = dfs_order(graph, agents)
        # n_coverage = plan_sga(ordered_agents)

        n_round, n_coverage = plan_dfs_sga(graph, agents)
        # for testing correctness of coverage
        # sim = Simulator(agents=agents, height=HEIGHT, width=WIDTH)
        # sim.simulate()
        # print(planner.compute_cost(agents))

        all_memory.append(np.round(psutil.Process().memory_info().rss / 1024 ** 2, 2))

        all_time.append(time.time() - start_time)
        all_n_round.append(n_round)
        all_n_coverage.append(n_coverage)
        # print('Coverage is :', coverage)
        # print('Comm Round is :', round)
        # plt.plot(range(SIM_STEPS+1), coverage)
        
    print('Aver Comm Round is :', np.mean(all_n_round)) 
    print('Aver Coverage is :', np.mean(all_n_coverage)) 
    print('Aver Time is :', np.mean(all_time)) 
    print('Aver Memory is :', np.mean(all_memory))



def plan_sga(agents):
    """
    Performs Sequential Greedy Assignment for DFS/as a baseline algorithm.
    :param agents: Set of agents to plan for.
    :param radius:
    :return: Number of observed points
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
    return len(observed_points)

def connected_connectivity_graph():
    """
    Construct connected connectivity graph for agents.
    """
    agents = [create_agent() for i in range(0, N_AGENTS)]
    graph = connectivity_graph(agents, COMM_RANGE)
    while not nx.is_connected(graph):
        agents = [create_agent() for i in range(0, N_AGENTS)]
        graph = connectivity_graph(agents, COMM_RANGE)
    return agents, graph

def connectivity_graph(agents, comm_range):
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
                G.add_edge(idx_i, idx_j)
    # for i in agents:
    #     G.add_node(i)
    # for idx_i, i in enumerate(agents):
    #     for idx_j, j in enumerate(agents):
    #         if np.linalg.norm((i.state[0] - j.state[0], i.state[1] - j.state[1])) < comm_range \
    #         and i != j:
    #             G.add_edge(i,j)
    return G

def dfs_order(graph, agents):
    def init(v, w):
        nonlocal agents
        graph.nodes[v]['done'] = True
        graph.nodes[v]['parent'] = w
        ordered_agents.append(agents[v])
        message(v)

    def message(v):
        nonlocal t
        order = len([w for w in graph.nodes if graph.nodes[w]['done']])
        nodes = [w for w in graph[v] if not graph.nodes[w]['done']]
        par = graph.nodes[v]['parent']
        if nodes:
            t += 1
            w = nodes[0]
            init(w, v)
        elif order != graph.number_of_nodes():
            t += 1
            message(par)

    for i in range(graph.number_of_nodes()):
        graph.nodes[i]['done'] = False
    t = 0
    ordered_agents = []
    init(0, -1)
    return t, ordered_agents

def plan_sga(ordered_agents):
    observed_points = set()
    for agent in ordered_agents:
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
    return len(observed_points)

def plan_dfs_sga(graph, agents):
    def init(v, w):
        nonlocal observed_points, agents
        graph.nodes[v]['done'] = True
        graph.nodes[v]['parent'] = w
        observed_points = local_sga(agents[v], observed_points)
        message(v)

    def message(v):
        nonlocal t, observed_points, agents
        order = len([w for w in graph.nodes if graph.nodes[w]['done']])
        nodes = [w for w in graph[v] if not graph.nodes[w]['done']]
        par = graph.nodes[v]['parent']
        if nodes:
            t += 1
            w = nodes[0]
            init(w, v)
        elif order != graph.number_of_nodes():
            t += 1
            message(par)
    
    def local_sga(agent, observed_points):
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
        return observed_points

    for i in range(graph.number_of_nodes()):
        graph.nodes[i]['done'] = False
    t = 0
    observed_points = set()
    init(0, -1)
    return t, len(observed_points)


def create_agent():
    x = np.random.choice(range(0, HEIGHT))
    y = np.random.choice(range(0, WIDTH))
    return Agent(state=(x, y), radius=RADIUS, height=HEIGHT, width=WIDTH, step=STEPSIZE, res=RES, color=np.random.rand(3))

def create_sparse_agent():
    locations = [(10,10), (20,20), (20,40), (30,80), (40,50), (50,70), (60,40), (70,80), (80,30), (90,50)]
    agents = []
    for i in range(len(locations)):
        agents.append(Agent(state=locations[i], radius=RADIUS, height=HEIGHT, width=WIDTH, step=STEPSIZE, res=RES, color=np.random.rand(3)))
    return agents

if __name__ == "__main__":
    main()
    # sparse_agents_RAG_DFS_compare()



#######################################################################
# DFS by Konda et al. (WIDTH=100, HEIGHT=100, N_AGENTS=40, RADIUS=4, res=1, 300 trials)
# 48.67 rounds, COMM_RANGE = 40
# 43.67 rounds, COMM_RANGE = 50
# 41.10 rounds, COMM_RANGE = 60
# 39.98 rounds, COMM_RANGE = 70
# 39.45 rounds, COMM_RANGE = 80
# 39.14 rounds, COMM_RANGE = 90
# 39.05 rounds, COMM_RANGE = 100