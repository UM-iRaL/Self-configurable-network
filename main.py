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
import math
from joblib import Parallel, delayed
import multiprocessing

N_STEPS = 18000  # Time steps
N_ROUNDS = 2000  # Action selection rounds
N_AGENTS = 60
N_TRIALS = 30
WIDTH = 100
HEIGHT = 100
RADIUS = 7
# COMM_RANGE = 60
COMM_RANGE_UP = 20
COMM_RANGE_LOW = 15
# STEPSIZE = 1
RES = 1
TF = 5.0 # time for a function evaluation
TC = 1.0 # time for communicating an action

# def main():
#     mem = max(memory_usage(proc=main1))
#     print("Maximum memory used: {} MB".format(mem))

def main():
    lower_neighborhood_size = 0
    upper_neighborhood_size = 5
    all_n_coverage_DFS = []
    all_n_coverage_Anaconda = [[] for i in range(lower_neighborhood_size, upper_neighborhood_size + 1)]

    for rnd_seed in range(N_TRIALS):
        np.random.seed(rnd_seed)
        agents, graph = connected_connectivity_graph()

        # DFS-SG
        tt, coverage = plan_dfs_sg(graph, agents)
        coverage = interpolate_arrays(tt, coverage, N_STEPS)
        all_n_coverage_DFS.append(coverage)

        # Anaconda
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(anaconda)(agents, max_neighborhood_size) for max_neighborhood_size in range(lower_neighborhood_size, upper_neighborhood_size + 1))
        for idx, result in enumerate(results):
            tt, coverage = result
            coverage = interpolate_arrays(tt, coverage, N_STEPS)
            all_n_coverage_Anaconda[idx].append(coverage)

    # parallelism for all MC trials does not work yet
    # # num_cores = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=N_TRIALS)(delayed(trial)(rnd_seed, lower_neighborhood_size, upper_neighborhood_size) for rnd_seed in range(N_TRIALS))
    # for rnd_seed, result in enumerate(results):
    #     all_n_coverage_DFS.append(result[0])
    #     for max_neighborhood_size in range(lower_neighborhood_size, upper_neighborhood_size + 1):
    #         all_n_coverage_Anaconda[max_neighborhood_size - lower_neighborhood_size].append(result[1][max_neighborhood_size - lower_neighborhood_size])

    np.save('TF_'+str(TF)+'_TC_'+str(TC)+'_DFS.npy', all_n_coverage_DFS)
    np.save('TF_'+str(TF)+'_TC_'+str(TC)+'_Anaconda.npy', all_n_coverage_Anaconda)

    plt.figure(1)
    mean = np.mean(np.array(all_n_coverage_DFS) / WIDTH / HEIGHT * 100, axis=0)
    std = np.std(np.array(all_n_coverage_DFS) / WIDTH / HEIGHT * 100, axis=0)
    plt.plot(range(0, N_STEPS+1), mean, label='DFS-SG', linewidth=2)
    plt.fill_between(range(0, N_STEPS+1), np.array(mean)-np.array(std), np.array(mean)+np.array(std), alpha=0.3)
    for idx in range(lower_neighborhood_size, upper_neighborhood_size + 1):
        n_coverage_Anaconda = all_n_coverage_Anaconda[idx - lower_neighborhood_size]
        mean = np.mean(np.array(n_coverage_Anaconda) / WIDTH / HEIGHT * 100, axis=0)
        std = np.std(np.array(n_coverage_Anaconda) / WIDTH / HEIGHT * 100, axis=0)
        plt.plot(range(0, N_STEPS+1), mean, label='Neighbors <= '+str(idx), linewidth=1.3)#, color=‘blue’)
        plt.fill_between(range(0, N_STEPS+1), np.array(mean)-np.array(std), np.array(mean)+np.array(std), alpha=0.3)
    plt.legend(prop={'family':'Times New Roman', 'size':10}, loc='lower right')#, frameon=False)
    plt.xticks(fontname='Times New Roman', fontsize=10)
    plt.yticks(fontname='Times New Roman', fontsize=10)
    plt.xlabel('Time (s)', fontname='Times New Roman', fontsize=10)
    plt.ylabel('Percentage of Covered Area (%)', fontname='Times New Roman', fontsize=10)
    plt.xlim(0, N_STEPS)
    plt.ylim(0, 60)
    plt.title('Tf = 1.0s, Tc = 1.0s', fontname='Times New Roman', fontsize=10)
    plt.show()

def trial(rnd_seed, lower_neighborhood_size, upper_neighborhood_size):
    np.random.seed(rnd_seed)
    agents, graph = connected_connectivity_graph()

    # DFS-SG
    tt_DFS, coverage_DFS = plan_dfs_sg(graph, agents)
    coverage_DFS = interpolate_arrays(tt_DFS, coverage_DFS, N_STEPS)

    # Anaconda
    # num_cores = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cores)(delayed(anaconda)(agents, max_neighborhood_size) for max_neighborhood_size in range(lower_neighborhood_size, upper_neighborhood_size + 1))
    # for idx, result in enumerate(results):
    #     tt, coverage = result
    #     coverage = interpolate_arrays(tt, coverage, N_STEPS)
    #     all_n_coverage_Anaconda[idx].append(coverage)
    #     print(idx)
    all_neighborhood_size_coverage_Anaconda = []
    # for max_neighborhood_size in range(lower_neighborhood_size, upper_neighborhood_size + 1):
    #     tt_Anaconda, coverage_Anaconda = anaconda(agents, max_neighborhood_size)
    #     coverage_Anaconda = interpolate_arrays(tt_Anaconda, coverage_Anaconda, N_STEPS)
    #     all_neighborhood_size_coverage_Anaconda.append(coverage_Anaconda)

    # nested parallelism does not work yet
    # if using multiprocessing
    def worker(max_neighborhood_size):
        return anaconda(agents, max_neighborhood_size)
    with mp.Pool(6) as p:
        sub_results = p.map(worker, range(lower_neighborhood_size, upper_neighborhood_size + 1))
    # if using joblib
    # sub_results = Parallel(n_jobs=upper_neighborhood_size - lower_neighborhood_size + 1)(delayed(anaconda)(agents, max_neighborhood_size) for max_neighborhood_size in range(lower_neighborhood_size, upper_neighborhood_size + 1))
    for idx, sub_result in enumerate(sub_results):
        tt, coverage = sub_result
        coverage = interpolate_arrays(tt, coverage, N_STEPS)
        all_neighborhood_size_coverage_Anaconda.append(coverage)
    return coverage_DFS, all_neighborhood_size_coverage_Anaconda


def main_Anaconda():
    '''
    1st simulation in CDC'24
    random connected graph
    '''
    all_n_coverage = []  
    lower_neighborhood_size = 1
    upper_neighborhood_size = 2

    for max_neighborhood_size in range(lower_neighborhood_size, upper_neighborhood_size + 1):
        n_coverage = []
        for rnd_seed in range(0,3):
            np.random.seed(rnd_seed)
            agents, _ = connected_connectivity_graph()
            tt, coverage = anaconda(agents, max_neighborhood_size)
            coverage = interpolate_arrays(tt, coverage, N_STEPS)
            n_coverage.append(coverage)
        all_n_coverage.append(n_coverage)

    plt.figure(1)
    for idx in range(upper_neighborhood_size - lower_neighborhood_size + 1):
        n_coverage = all_n_coverage[idx]
        mean = np.mean(n_coverage, axis=0)
        std = np.std(n_coverage, axis=0)
        plt.plot(range(0, N_STEPS+1), mean, label=idx+lower_neighborhood_size, )#, color='blue', linewidth=2)
        plt.fill_between(range(0, N_STEPS+1), np.array(mean)-np.array(std), np.array(mean)+np.array(std), alpha=0.3)
    plt.legend(prop={'family':'Times New Roman', 'size':20}, loc='lower right')#, frameon=False)
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    plt.xlabel("Time (s)", fontname="Times New Roman", fontsize=20)
    plt.ylabel("Covered Area", fontname="Times New Roman", fontsize=20)
    plt.show()

def anaconda(agents, max_neighborhood_size):
    tt = 0
    all_tt = []
    n_coverage = []
    for i in agents:
        # i.in_neighbor_candidates = set()
        for j in agents:
            if np.linalg.norm((i.state[0] - j.state[0], i.state[1] - j.state[1])) < i.comm_range and i != j:
                i.in_neighbor_candidates.append(j)
        i.in_neighborhood_candidate_size = len(i.in_neighbor_candidates)
        # i.in_neighborhood_size = math.ceil(i.in_neighborhood_candidate_size * 1 / 3)
        i.in_neighborhood_size = min(max_neighborhood_size, math.ceil(i.in_neighborhood_candidate_size))

        if i.in_neighborhood_candidate_size == 0 or i.in_neighborhood_size == 0:
            i.neighbor_select_flag = False
            continue

    # for counting the time of each round
    max_agent_neighborhood_size = max([i.in_neighborhood_size for i in agents])
    max_agent_n_action = max([i.n_actions for i in agents])
    # n_rounds = N_ROUNDS
    n_rounds = math.ceil(N_STEPS / (TF * max_agent_n_action + TC * 1))

    for i in agents:
        if i.neighbor_select_flag == False:
            continue
        i.eta_nei = np.sqrt(2 * math.log(i.in_neighborhood_candidate_size) / i.in_neighborhood_candidate_size / n_rounds) # learning rate for selecting neighbors
        i.gamma_nei = i.eta_nei / 2 # bias for selecting neighbors
        i.neighbor_weight = [[[1.0 / i.in_neighborhood_candidate_size for ii in range(i.in_neighborhood_candidate_size)] for j in range(i.in_neighborhood_size)] for t in range(n_rounds+1)]  # weights of neighbor candidates for all EXP3-IX
        i.neighbor_loss = [[[0.0 for ii in range(i.in_neighborhood_candidate_size)] for j in range(i.in_neighborhood_size)] for t in range(n_rounds+1)]  # weights of neighbor candidates for all EXP3-IX

    for t in range(n_rounds):

        observed_points = set()

        # select an action for each agent
        for agent in agents:
            # output action probability distribution by FSF*
            # agent.get_action_prob_dist(t)

            # sample the next action
            # print(agent.action_prob_dist[t])
            # next_action_index = np.random.choice(agent.action_indices, 1, p=agent.action_prob_dist[t])[0]
            next_action_index = np.random.choice(agent.action_indices, 1, p=agent.action_weight[t])[0]
            agent.next_action_index = next_action_index
            # agent.action_hist.append(next_action_index)

            # get the next placement
            # agent.get_next_placement()
            agent.next_placement = agent.motion_model(agent.state, agent.actions[agent.next_action_index])

            # count the observed points by all agents
            observed_points = observed_points.union(agent.get_observations(agent.next_placement))


        # select neighbors for each agent
        for agent in agents:

            if agent.neighbor_select_flag == False:
                continue

            for idx in range(agent.in_neighborhood_size):
                next_neighbor_index = np.random.choice(np.array(range(agent.in_neighborhood_candidate_size)), 1, p=agent.neighbor_weight[t][idx])[0]
                next_neighbor = agent.in_neighbor_candidates[next_neighbor_index]
                # agent.next_action_index = next_action_index
                # agent.neighbor_hist.append(next_neighbor_index) 

                neighbors_selected = set(agent.in_neighbors[t])
                agent.get_neighbor_losses(t, idx, neighbors_selected, next_neighbor_index)
                agent.update_neighbor_weights(t, idx)    

                agent.in_neighbors[t].append(next_neighbor)


        for agent in agents:
            # apply the next action 
            # agent.apply_next_action()
            # agent.traj.append(agent.state)

            # get loss vector based on neighbors' next placements
            agent.get_action_losses(t)

            # update distribution
            agent.update_action_weights(t)
        
        if tt + math.ceil(TF * (max_agent_n_action + 2 * max_agent_neighborhood_size + 1) + TC * 1) > N_STEPS:
            print('Time is up! t =', t)
            break

        n_coverage.append(len(observed_points))
        all_tt.append(tt)

        tt += math.ceil(TF * (max_agent_n_action + 2 * max_agent_neighborhood_size + 1) + TC * 1)

    return all_tt, n_coverage


def main_dfs_sg():
    '''
    random connected graph
    '''
    all_n_round = []    
    all_n_coverage = []
    all_time = []
    all_memory = []
    for rnd_seed in range(0, 20):
        np.random.seed(rnd_seed)
        # agents = [create_agent() for i in range(0, N_AGENTS)]
        # agents = create_sparse_agent()
        # graph = connectivity_graph(agents, comm_range=COMM_RANGE)
        agents, graph = connected_connectivity_graph()

        # start_time = time.time()

        tt, coverage = plan_dfs_sg(graph, agents)
        coverage = interpolate_arrays(tt, coverage, N_STEPS)

        # all_memory.append(np.round(psutil.Process().memory_info().rss / 1024 ** 2, 2))

        # all_time.append(time.time() - start_time)
        # all_time.append(time)
        all_n_coverage.append(coverage)
        # print('Coverage is :', coverage)
        # print('Comm Round is :', round)
        # plt.plot(range(SIM_STEPS+1), coverage)
    print('All Coverage is :', all_n_coverage) 
    print('All Time is :', range(0, N_STEPS+1)) 
    # print('Aver Comm Round is :', np.mean(all_n_round)) 
    # print('Aver Coverage is :', np.mean(all_n_coverage)) 
    # print('Aver Time is :', np.mean(all_time)) 
    # print('Aver Memory is :', np.mean(all_memory))

    plt.figure(1)
    mean = np.mean(all_n_coverage, axis=0)
    std = np.std(all_n_coverage, axis=0)
    plt.plot(range(0, N_STEPS+1), mean, label="DFS-SG")#, color='blue', linewidth=2)
    plt.fill_between(range(0, N_STEPS+1), np.array(mean)-np.array(std), np.array(mean)+np.array(std), alpha=0.3)
    plt.legend(prop={'family':'Times New Roman', 'size':20}, loc='lower right')#, frameon=False)
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    plt.xlabel("Time (s)", fontname="Times New Roman", fontsize=20)
    plt.ylabel("Covered Area", fontname="Times New Roman", fontsize=20)
    plt.show()

def sg():
    """
    Performs Sequential Greedy Assignment as a baseline algorithm.
    :param agents: Set of agents to plan for.
    :param radius:
    :return: None
    """
    all_n_coverage = []
    all_n_round = []
    for rnd_seed in range(0, 10):
        np.random.seed(rnd_seed)
        agents = create_random_agent()
        observed_points = set()
        for agent in agents:
            # best_action = []
            best_cost = -1
            best_set = set()
            for succ in agent.get_successors():
                succ_best_cost = len(observed_points.union(agent.get_observations(succ)))
                if succ_best_cost > best_cost:
                    # best_action = action
                    best_cost = succ_best_cost
                    best_set = agent.get_observations(succ)
            observed_points = observed_points.union(best_set)
            # agent.set_next_action(best_action)
        all_n_coverage.append(len(observed_points))
    print(all_n_coverage)

def plan_dfs_sg(graph, agents):
    def init(v, w):
        nonlocal t, observed_points, agents, message_length
        graph.nodes[v]['done'] = True
        graph.nodes[v]['parent'] = w
        observed_points = local_sg(agents[v], observed_points)
        t += TF * agents[v].n_actions # evaluate all available actions
        message_length += 1
        all_time.append(math.ceil(t))
        all_n_coverage.append(len(observed_points))
        message(v)

    def message(v):
        nonlocal t, observed_points, agents, message_length
        order = len([w for w in graph.nodes if graph.nodes[w]['done']])
        nodes = [w for w in graph[v] if not graph.nodes[w]['done']]
        par = graph.nodes[v]['parent']
        if nodes:
            t += TC * message_length
            w = nodes[0]
            init(w, v)
        elif order != graph.number_of_nodes():
            t += TC * message_length
            message(par)
    
    def local_sg(agent, observed_points):
        # best_action = []
        best_cost = -1
        best_set = set()
        for succ in agent.get_successors():
            succ_best_cost = len(observed_points.union(agent.get_observations(succ)))
            if succ_best_cost > best_cost:
                # best_action = action
                best_cost = succ_best_cost
                best_set = agent.get_observations(succ)
        observed_points = observed_points.union(best_set)
        # agent.set_next_action(best_action)
        return observed_points

    for i in range(graph.number_of_nodes()):
        graph.nodes[i]['done'] = False
    t = 0
    message_length = 0
    observed_points = set()
    all_time = [0]
    all_n_coverage = [0]
    init(0, -1)
    return all_time, all_n_coverage

def create_agent():
    x = np.random.choice(range(0, HEIGHT))
    y = np.random.choice(range(0, WIDTH))
    r = np.random.choice(range(COMM_RANGE_LOW, COMM_RANGE_UP))
    return Agent(state=(x, y), radius=RADIUS, comm_range=r, height=HEIGHT, width=WIDTH, res=RES, n_time_step=N_STEPS, color='none')#color=np.random.rand(3))

def create_sparse_agent():
    locations = [(10,10), (20,20), (20,40), (30,80), (40,50), (50,70), (60,40), (70,80), (80,30), (90,50)]
    agents = []
    for i in range(len(locations)):
        agents.append(Agent(state=locations[i], radius=RADIUS, height=HEIGHT, width=WIDTH, res=RES, n_time_step=N_STEPS, color=np.random.rand(3)))
    return agents

def create_random_agent():
    locations = [(np.random.choice(range(WIDTH), 1)[0], np.random.choice(range(HEIGHT), 1)[0]) for i in range(N_AGENTS)]
    agents = []
    for i in range(len(locations)):
        agents.append(Agent(state=locations[i], radius=RADIUS, height=HEIGHT, width=WIDTH, res=RES, n_time_step=N_STEPS, color=np.random.rand(3)))
    return agents

def create_dense_agent():
    # locations = [(10,10), (20,20), (20,40), (30,80), (40,50), (50,70), (60,40), (70,80), (80,30), (90,50)]
    locations = [(10,10), (10,30), (20,40), (40,40), (40,20), (70,40), (85,30), (25,60), (60,60), (80,60), (70,20), (60,95), (30,80), (40,70), (50,75), (60,30), (70,80), (90,50)]
    agents = []
    for i in range(len(locations)):
        agents.append(Agent(state=locations[i], radius=RADIUS, height=HEIGHT, width=WIDTH, res=RES, n_time_step=N_STEPS, color=np.random.rand(3)))
    return agents

def connected_connectivity_graph():
    """
    Construct connected connectivity graph for agents.
    """
    agents = [create_agent() for i in range(0, N_AGENTS)]
    graph = connectivity_graph(agents)
    while not nx.is_connected(graph):
        agents = [create_agent() for i in range(0, N_AGENTS)]
        graph = connectivity_graph(agents)
    return agents, graph

def connectivity_graph(agents):
    """
    Construct connectivity graph for agents.
    """
    G = nx.Graph()
    for idx_i, i in enumerate(agents):
        G.add_node(idx_i)
    for idx_i, i in enumerate(agents):
        for idx_j, j in enumerate(agents):
            if np.linalg.norm((i.state[0] - j.state[0], i.state[1] - j.state[1])) < i.comm_range \
            and i != j:
                G.add_edge(idx_i, idx_j)
    return G

def interpolate_arrays(input_array1, input_array2, scalar):
    # Create the first output array as a range from 0 to scalar, inclusive
    output_array1 = np.arange(scalar + 1)
    
    # Initialize the second output array with zeros
    output_array2 = np.zeros(scalar + 1)
    
    # Loop through the first input array to determine the ranges for interpolation
    for i in range(len(input_array1) - 1):
        start_idx = input_array1[i]
        end_idx = input_array1[i + 1]
        
        # Fill in the values in the output array
        output_array2[start_idx:end_idx] = input_array2[i]
        
    # Ensure the last value fills to the end
    output_array2[input_array1[-1]:] = input_array2[-1]
    
    return output_array2

if __name__ == "__main__":
    main()
    # main_dfs_sg()
    # main_Anaconda()
    # RAG_diff_comm_range()
    # sparse_agents_RAG_DFS_compare()
