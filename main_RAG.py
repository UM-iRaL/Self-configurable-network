import simulator
from agent.agent import Agent
from simulator.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
from planner.planner import Planner
import time
import psutil
# from memory_profiler import memory_usage
import networkx as nx

SIM_STEPS = 1
N_AGENTS = 100
WIDTH = 300
HEIGHT = 30
RADIUS = 15
COMM_RANGE = 20
STEPSIZE = 1
RES = 1

# np.random.seed(1)

# def main():
#     mem = max(memory_usage(proc=main1))
#     print("Maximum memory used: {} MB".format(mem))

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
    1st simulation in CDC'22
    random connected graph
    '''
    all_n_round = []   
    all_n_coverage = []  
    all_time = []  
    all_memory = []
    for rnd_seed in range(1, 51):
        np.random.seed(rnd_seed)
        # agents = [create_agent() for i in range(0, N_AGENTS)]
        # agents = create_sparse_agent()
        agents, _ = connected_connectivity_graph()
        sim = Simulator(agents=agents, height=HEIGHT, width=WIDTH)
        planner = Planner()
        sim.draw()
        plt.savefig("map.png")

        start_time = time.time()
        n_coverage, n_round = planner.plan_rag(agents, COMM_RANGE)

        # for testing correctness of coverage
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

    # plt.boxplot(np.array(all_n_round))
    # plt.hist(np.array(all_n_round))
    # plt.show()

def RAG_diff_comm_range():
    '''
    2nd simulation in CDC'22
    '''
    all_n_round = []   
    all_n_coverage = []  
    all_time = []  
    all_memory = []
    for communication_range in range(1, 51):
        np.random.seed(1)
        agents = [create_agent() for i in range(0, N_AGENTS)]
        sim = Simulator(agents=agents, height=HEIGHT, width=WIDTH)
        planner = Planner()

        start_time = time.time()

        n_coverage, n_round = planner.plan_rag(agents, communication_range)
        all_memory.append(np.round(psutil.Process().memory_info().rss / 1024 ** 2, 2))
        all_time.append(np.round(time.time() - start_time, 4))
        all_n_round.append(n_round)
        all_n_coverage.append(n_coverage)
        # all_memory.append(np.round(max(memory_usage((planner.plan_rag, (agents, communication_range))))), 3)
        # plt.plot(range(SIM_STEPS+1), coverage)
        
    print('Comm Round is :', all_n_round) 
    print('Coverage is :', all_n_coverage)
    print('Time is :', all_time)
    print('Memory is :', all_memory)

def trial():
    agents = [create_agent() for i in range(0, N_AGENTS)]
    sim = Simulator(agents=agents, height=HEIGHT, width=WIDTH)
    planner = Planner()

    # print('Current Coverage is :', planner.compute_cost(agents))
    coverage = [] #[planner.compute_cost(agents)]
    round = []
    comp_time = []
    for t in range(0, SIM_STEPS):
        # sim.draw()
        # plt.pause(0.5)

        start_time = time.time()
        n_round = planner.plan_rag(agents, comm_range=COMM_RANGE)
        # planner.plan(agents, n_iters=T)
        # planner.plan_sga(agents)

        # sim.simulate()
        comp_time.append(time.time() - start_time)
        # sim.draw()
        # print('Current Coverage is :', planner.compute_cost(agents))
        # print('Current Comm Round is :', n_round)
        coverage.append(planner.compute_cost(agents))
        round.append(n_round)
        

    # print('Simulation time: ', time.time() - start_time)

    return coverage, round, comp_time

def create_agent():
    x = np.random.choice(range(0, HEIGHT))
    y = np.random.choice(range(0, WIDTH))
    return Agent(state=(x, y), radius=RADIUS, height=HEIGHT, width=WIDTH, step=STEPSIZE, res=RES, color='none')#color=np.random.rand(3))

def create_sparse_agent():
    locations = [(10,10), (20,20), (20,40), (30,80), (40,50), (50,70), (60,40), (70,80), (80,30), (90,50)]
    agents = []
    for i in range(len(locations)):
        agents.append(Agent(state=locations[i], radius=RADIUS, height=HEIGHT, width=WIDTH, step=STEPSIZE, res=RES, color=np.random.rand(3)))
    return agents

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

def sparse_agents_RAG_DFS_compare():
    '''

    '''
    all_n_round = []   
    all_n_coverage = []  
    all_time = []  
    all_memory = []
    for rnd_seed in range(1, 2):
        # np.random.seed(rnd_seed)
        # agents = [create_agent() for i in range(0, N_AGENTS)]
        agents = create_sparse_agent()
        graph = connectivity_graph(agents, comm_range=15)
        sim = Simulator(agents=agents, height=HEIGHT, width=WIDTH)
        planner = Planner()
        sim.draw()
        plt.savefig("map.png")

        start_time = time.time()
        n_coverage, n_round = planner.plan_rag(agents, comm_range=15)

        all_memory.append(np.round(psutil.Process().memory_info().rss / 1024 ** 2, 2))

        all_time.append(time.time() - start_time)
        all_n_round.append(n_round)
        all_n_coverage.append(n_coverage)
        
    print('Aver Comm Round is :', np.mean(all_n_round)) 
    print('Aver Coverage is :', np.mean(all_n_coverage)) 
    print('Aver Time is :', np.mean(all_time)) 
    print('Aver Memory is :', np.mean(all_memory))



if __name__ == "__main__":
    main()
    # RAG_diff_comm_range()
    # sparse_agents_RAG_DFS_compare()
