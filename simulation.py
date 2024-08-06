import numpy as np
import time
import networkx as nx
from collections import defaultdict, deque
from collections import OrderedDict



def update_current(current, dependence_dict, not_fixed):
    updated_current = []
    for curr in current:
        current_dep = dependence_dict[curr]
        for node in current_dep:
            if node in not_fixed:
                updated_current.append(node)
                updated_current.remove(curr)
                break
            else:
                updated_current.append(curr)
    return updated_current


def dependency_information_hfg(dependency_matrix):
    # Get the dimensions of the dependency matrix
    rows, cols = dependency_matrix.shape
    # Dictionary to store dependencies for each column
    dependency_dict = {}

    # Loop through each column
    for j in range(cols):
        # Find rows where dependency is present (value equals 1)
        dependencies = [i for i in range(rows) if dependency_matrix[i, j] == 1]
        # Store dependencies for the current column in the dictionary
        dependency_dict[j] = dependencies

    # Return the dictionary containing dependency information
    return dependency_dict


def get_array(file):
    """
    # Generate the adjacency matrix
    Input: Graphml of the heterofunctional graph
    Returns adjacency matrix and the node names
    """
    A = nx.adjacency_matrix(file)

    # Convert the adjacency matrix to a dense NumPy array
    adjacency_matrix = A.toarray()

    # Get the list of nodes in the same order as the rows/columns of the adjacency matrix
    nodes = list(file.nodes())
    return adjacency_matrix, nodes




def change_state_damage(initial_state, next_damage_array):
    """
    this follows the rule that if damage is 1, then the functionality state is 0, 1 otherwise
    :param initial_state: initial state of the system
    :param next_damage_array: damage after actions have been taken
    :return: new intermediate state of the system
    """
    length = len(initial_state)
    intermediate_array = np.zeros([len(initial_state), 1])

    for i in range(length):
        intermediate_array[i][0] = 0 if next_damage_array[i][0] == 1 else 1

    return intermediate_array


def apply_hfg_policy(intermediate_state, policy_matrix):
    """
    We use an AND gate to get the final state of each functionality.
    :param intermediate_state: gives an intermediate state of each functionality in the system
    :param policy_matrix: hetero-functional graph
    :return:
    """
    original_array = np.array(intermediate_state)
    final_state = [[0] for _ in original_array]
    # print(final_state)
    init_length = len(intermediate_state)

    for col_index in range(init_length):
        # print("col_index",col_index)
        one_indices = [(row_index, col_index) for row_index in range(init_length) if
                       policy_matrix[row_index][col_index] == 1]
        # print("one_indices",one_indices)

        set_index = set([element for tup in one_indices for element in tup])
        # print("set_index",set_index)
        if len(one_indices) > 0:
            # print("col_index",col_index)

            # final_state[col_index][0] = [num * initial_state[item] for item in set_index][-1]
            # print("int",intermediate_state)
            final_state[col_index][0] = int(all([intermediate_state[sublist][0] for sublist in set_index]))
            # print("col_index",col_index,final_state)
            # print("int",intermediate_state)
        else:
            final_state[col_index][0] = intermediate_state[col_index][0]
    return final_state


def flatten(alist):
    newlist = [i[0] for i in alist]
    return newlist


def get_action_rule(hfg):
    """
    this is a topological ordering
    """

    def dfs(node):
        visited[node] = True
        for neighbor in range(len(hfg)):
            if hfg[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor)
        action_rule.insert(0, node)

    num_nodes = len(hfg)
    visited = [False] * num_nodes
    action_rule = []

    for node in range(num_nodes):
        if not visited[node]:
            dfs(node)

    return action_rule

def update_graph(alpha,graph,consumer_path_dict):

    if alpha == "s":
        svs = defaultdict(int)

        # Iterate through consumer_dict
        for key, numbers in consumer_path_dict.items():
            # Iterate through nodes in the graph
            for node in graph.nodes(data=True):
                node_attrs = node[1]


                # Check if 'resource' attribute of node matches any number in consumer_dict
                if node_attrs['index'] in numbers and node_attrs['community']:
                    # Update max Criticality for the current key if found
                    svs[key] = max(svs[key], node_attrs['SVS'])

        # Correct code
        for node_id, node_attrs in graph.nodes(data=True):
            # Check if 'index' attribute of node exists as a key in max_criticalities
            if node_attrs['index'] in svs:
                # Update 'Criticality' attribute of the node with the maximum value
                node_attrs['Criticality'] = svs[node_attrs['index']]


    elif alpha == "c":
        max_criticalities = defaultdict(int)

        # Iterate through consumer_dict
        for key, numbers in consumer_path_dict.items():
            # Iterate through nodes in the graph
            for node in graph.nodes(data=True):
                node_attrs = node[1]
                # print(node_attrs)

                # Check if 'resource' attribute of node matches any number in consumer_dict
                if node_attrs['index'] in numbers and node_attrs['community']:
                    # Update max Criticality for the current key if found
                    max_criticalities[key] = max(max_criticalities[key], node_attrs['Criticality'])

        # Correct code
        for node_id, node_attrs in graph.nodes(data=True):
            # Check if 'index' attribute of node exists as a key in max_criticalities
            if node_attrs['index'] in max_criticalities:
                # Update 'Criticality' attribute of the node with the maximum value
                node_attrs['Criticality'] = max_criticalities[node_attrs['index']]



    elif alpha == "sc":

        combined_values = defaultdict(lambda: {'max_svs': 0, 'max_criticality': 0})

        # Iterate through consumer_path_dict

        for key, numbers in consumer_path_dict.items():

            # Iterate through nodes in the graph

            for node_id, node_attrs in graph.nodes(data=True):

                # Check if 'index' attribute of node matches any number in consumer_path_dict

                # Add additional condition 'community' here if necessary

                if node_attrs.get('index') in numbers and node_attrs.get('community'):
                    # Update max SVS for the current key if found

                    combined_values[key]['max_svs'] = max(combined_values[key]['max_svs'], node_attrs.get('SVS', 0))

                    # Update max Criticality for the current key if found

                    combined_values[key]['max_criticality'] = max(combined_values[key]['max_criticality'],
                                                                  node_attrs.get('Criticality', 0))

        # Update 'SVS' and 'Criticality' attributes of nodes in the graph based on combined max SVS and max Criticality

        for node_id, node_attrs in graph.nodes(data=True):

            if node_attrs.get('index') in combined_values:
                node_attrs['SVS'] = combined_values[node_attrs.get('index')]['max_svs']

                node_attrs['Criticality'] = combined_values[node_attrs.get('index')]['max_criticality']

    return graph
def get_action_rule_metrics(graph, start_node, node_metrics,alpha,consumer_path_dict):
    """
    Uses BFS
    """

    graph = update_graph(alpha, graph, consumer_path_dict)
    visited = set()
    sorted_nodes = []
    queue = deque([(start_node, 0)])  # (node, level)

    level_dictionary = {}
    while queue:
        node, level = queue.popleft()
        if level not in level_dictionary.keys():
            level_dictionary[level] = []
        level_dictionary[level].append(node)

        if node not in visited:
            visited.add(node)
            if len(sorted_nodes) <= level:
                sorted_nodes.append([])
            sorted_nodes[level].append(node)

            neighbors = sorted(graph[node], key=lambda x: graph.nodes()[x][node_metrics],
                               reverse=True)  # Sort neighbors based on metrics in descending order
            for neighbor in neighbors:
                queue.append((neighbor, level + 1))

    sorted_vals = [node for level_nodes in sorted_nodes for node in level_nodes]

    new_results = []
    for i, v in level_dictionary.items():
        neighbors = sorted(v, key=lambda x: graph.nodes[x][node_metrics], reverse=True)
        for n in neighbors:
            new_results.append(n)

    return [list(graph.nodes()).index(node) for node in new_results]


def get_action_rule_metrics_weighted(graph, start_node, metric1, metric2, weights,alpha, consumer_path_dict):
    """
    Uses BFS to sort nodes based on combined weighted metrics
    """
    graph = update_graph(alpha, graph, consumer_path_dict)
    visited = set()
    sorted_nodes = []
    queue = deque([(start_node, 0)])  # (node, level)

    level_dictionary = {}
    while queue:
        node, level = queue.popleft()
        if level not in level_dictionary.keys():
            level_dictionary[level] = []
        level_dictionary[level].append(node)
        # print(node,level)

        if node not in visited:
            visited.add(node)
            if len(sorted_nodes) <= level:
                sorted_nodes.append([])
            sorted_nodes[level].append(node)

            # Combine metrics using weights
            neighbors = sorted(graph[node],
                               key=lambda x: weights[0] * graph.nodes[x][metric1] + weights[1] * graph.nodes[x][
                                   metric2],
                               reverse=True)  # Sort neighbors based on combined weighted metrics in descending order
            for neighbor in neighbors:
                queue.append((neighbor, level + 1))

    #sorted_vals = [node for level_nodes in sorted_nodes for node in level_nodes]
    new_results = []
    for i, v in level_dictionary.items():
        # combined_metric = weights[0] * graph.nodes[node][metric1] + weights[1] * graph.nodes[node][metric2]
        neighbors = sorted(v, key=lambda x: weights[0] * graph.nodes[x][metric1] + weights[1] * graph.nodes[x][metric2],
                           reverse=True)
        for n in neighbors:
            new_results.append(n)

    return [list(graph.nodes()).index(node) for node in new_results]


def get_current(action_rule, num_crew, damage):
    current = []
    for action in action_rule:
        # print("action in action_rule",action)
        if damage[action] == 1:
            current.append(action)
            if len(current) >= num_crew:
                break  # Break the loop once you have enough crew members
    return current


def change_to_opposite(lst):
    return [1 - x for x in lst]



def get_initial_damage(initial_state_resources, heterofunctional_adjacency_matrix):
    failed = []
    for init in initial_state_resources.keys():
        if initial_state_resources[init] == 'fail':
            failed.append(init)

    functionalities = list(heterofunctional_adjacency_matrix.index)
    initial_damage = []  # Initialize initial_damage list

    for func in functionalities:
        func_split = func.split()
        for res in failed:
            if res in func_split:
                initial_damage.append(1)
                break
        else:
            initial_damage.append(0)

    return initial_damage


def get_transition_matrix(predetermined_steps, lambda_parameter=0.5):
    """
    User defines the transition probability matrix
    :param predetermined_steps:
    :param lambda_parameter = 0.5  # Adjust the lambda parameter for the exponential distribution
    :return:
    """

    # Define constant probabilities for the other transitions
    transition_matrix = {}
    transition_matrix[0] = 1  # 0.0
    transition_matrix[1] = 0.999999  # 0.999999999999

    # Define the time-dependent exponential distribution for transition_matrix[1, 1, 0]
    transition_matrix[2] = {}

    for t in range(predetermined_steps):
        # print("t",t)
        time_since_last_transition = t  # For simplicity, assuming time starts at 0
        probability = 1 - np.exp(-lambda_parameter * time_since_last_transition)
        transition_matrix[2][t] = probability
    # print(transition_matrix)
    return transition_matrix


def find_cycles(matrix):
    visited = set()
    cycles = []

    def dfs(node, start_node, cycle):
        if node == start_node and len(cycle) > 1:
            cycles.append(cycle[:])  # Append a copy of the cycle
            return

        if node in cycle or node in visited:
            return

        visited.add(node)
        cycle.append(node)

        for neighbor, connected in enumerate(matrix[node]):
            if connected:
                dfs(neighbor, start_node, cycle)

        cycle.pop()
        visited.remove(node)

    for start_node in range(len(matrix)):
        dfs(start_node, start_node, [])

    if cycles:
        unique_cycles = set(tuple(sorted(cycle)) for cycle in cycles)
        # Convert back to list of lists
        unique_cycles = [list(cycle) for cycle in unique_cycles]
        return unique_cycles
    else:
        return None


def update_current_recursive(current, dependence_info_dict, not_fixed, updated_current=None):
    if updated_current is None:
        updated_current = OrderedDict()

    for node in current:
        if node not in updated_current:
            parents = dependence_info_dict.get(node, [])
            for parent in parents:
                if parent in not_fixed and parent not in updated_current:
                    updated_current[parent] = None  # Use None as placeholder
                    update_current_recursive([parent], dependence_info_dict, not_fixed, updated_current)

            updated_current[node] = None  # Mark the current node as processed
    # print("current,updated_currentrecursive",current,updated_current)
    return list(updated_current.keys())


def find_cycles_containing_node(node, cycles):
    cycles_containing_node = []
    for cycle in cycles:
        if node in cycle:
            # print("node,cycle",node,cycle)
            cycles_containing_node.extend(cycle)
    return cycles_containing_node


def modify_hfg(hfg, cycles):
    # Iterate through the cycles and disconnect edges
    for cycle in cycles:
        for i in range(len(cycle)):
            current_node = cycle[i]
            next_node = cycle[(i + 1) % len(cycle)]  # Wrap around to the first node if last node in the cycle
            hfg[current_node, next_node] = 0  # Disconnect the edge

    return hfg




def get_dependent_nodes(graph, node, community):
    """
    Get the dependent nodes of a given node in a directed graph using DFS traversal.

    Args:
    - graph: NetworkX directed graph
    - node: Node for which dependent nodes are to be found

    Returns:
    - dependent_nodes: Set of dependent nodes
    """
    dependent_nodes = set()  # Initialize set to store dependent nodes
    visited = set()  # Initialize set to store visited nodes during traversal

    node = list(graph.nodes())[node]
    dependent_comm = []

    def dfs(current_node):
        """
        Depth-first search (DFS) traversal to find dependent nodes.

        Args:
        - current_node: Current node being visited in DFS traversal
        """
        # Mark current node as visited
        visited.add(current_node)

        # Explore neighbors of current node
        for neighbor in graph.successors(current_node):
            # If neighbor is not visited, recursively perform DFS
            if neighbor not in visited:
                dfs(neighbor)
                # Add neighbor to dependent nodes
                dependent_nodes.add(neighbor)

    # Perform DFS traversal from the given node
    dfs(node)

    for i in dependent_nodes:
        # check if in community

        if community in graph.nodes()[i].keys():
            if graph.nodes()[i][community] == 1:
                dependent_comm.append(i)

    dependent_nodes_ind = [list(graph.nodes()).index(nd) for nd in dependent_comm]
    return dependent_nodes_ind


def generate_dependency_dict(dependency_matrix):
    dependency_dict = {}
    num_nodes = dependency_matrix.shape[0]

    for node in range(num_nodes):
        children = []
        for child, dependency in enumerate(dependency_matrix[node]):
            if dependency == 1:
                children.append(child)
        dependency_dict[node] = children

    return dependency_dict




def get_power_action(action_rule, all_nodes_power):
    action = []
    for act in action_rule:
        # print("act",act)
        if act in all_nodes_power and (act not in action):
            action.append(act)
    return action


def get_new_hfg(array, rows_cols_to_maintain, last_node):
    # Create a new 9x9 array filled with zeros
    extracted_array = np.zeros((last_node, last_node))

    # Copy the original values for specified rows and columns into the new array
    for i in rows_cols_to_maintain:
        for j in rows_cols_to_maintain:
            extracted_array[i, j] = array[i, j]

    return extracted_array


def get_source_node_by_index(graph, index):
    for node in graph.nodes():
        if graph.nodes[node]['index'] == index:
            return node

def unique_actions(input_paths):
    # This set will keep track of unique actions to avoid duplicates
    seen = set()
    # This list will store the final sequence of actions
    output = []

    # Iterate through each path list in the input_paths
    for path in input_paths:
        # Iterate through each action in the path
        for action in path:
            # Only add the action if it has not been seen before
            if action not in seen:
                seen.add(action)
                output.append(action)

    return output


def find_paths_with_same_type_and_community(graph, source_nodes):
    paths_dict = {}

    # Perform BFS traversal from each source node
    print("source_nodes",source_nodes)
    for source_node_index in source_nodes:
        source_node = get_source_node_by_index(graph, source_node_index)
        visited = set()  # Set to keep track of visited nodes
        queue = deque([(source_node, [source_node])])  # Queue for BFS traversal
        paths = []
        while queue:
            current_node, current_path = queue.popleft()
            visited.add(current_node)
            # Check if the current node has the 'type' attribute
            if 'type' not in graph.nodes[current_node]:
                continue  # Skip if 'type' attribute does not exist

            # Check if the current node has the same type as the source node
            if graph.nodes[current_node]['type'] != graph.nodes[source_node]['type']:
                continue  # Skip if the types are different

            # Check if the current node has the 'community' attribute set to True
            if graph.nodes[current_node].get('community', False):
                # Add the current path to the list of paths
                paths.append(current_path)
                continue

            # Enqueue neighboring nodes for traversal
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    new_path = current_path + [neighbor]
                    queue.append((neighbor, new_path))

        paths_dict[source_node_index] = paths

    return paths_dict

#consumer_path_dict {34: {34, 35, 40, 41, 42, 43, 44, 45, 46, 47}, 36: {36, 37, 48, 49, 50, 51, 52, 53}}


def simulate_damage_evolution_water(damage, initial_states, hfg, num_crew, action_rule, probs,
                                    consumer_dicts,consumer_path_dict,failed_indices,water_storages,criticality):
    """

    :param damage:
    :param initial_states:
    :param hfg:
    :param num_crew:
    :param action_rule:
    :param probs:
    :param consumer_dicts:
    :param demand_list: This could be Demand not served in Power
    :return:
    """
    # dns_dict = {}
    # for key, each in zip(consumer_dicts.keys(), demand_list):
    #     dns_dict[key] = each

    time_step = 1
    next_damage = damage.copy()
    new_states = [0] * ((action_rule[-1]) + 1)

    calculate_state = []
    calculate_state.append(change_to_opposite(list(damage)))
    cycles = find_cycles(hfg)

    not_fixed = [i for i, v in enumerate(next_damage) if v == 1]

    while new_states != initial_states:
        # for i in not_fixed:
        #     damage_times[i] += 1

        current = get_current(action_rule, num_crew, next_damage)

        time_stamp = time_step

        for i in action_rule:
            if i in current:
                # repair_queue_times[i] += 1
                # if next_damage[i] == 1 and repair_times[i] is None:
                #     repair_times[i] = time_step  # Set repair start time if node is being repaired for the first time
                # Perform repair based on transition probabilities
                random_number = np.random.random()
                last_key = list(probs[2].keys())[-1]
                if time_step > last_key:
                    time_stamp = last_key
                prob_transition = probs[2].get(time_stamp)
                prob_transition = 1 - prob_transition
                if random_number >= prob_transition:
                    next_damage[i] = 0
                else:
                    next_damage[i] = 1
            else:
                if damage[i] == 0:
                    random_number = np.random.random()
                    prob_transition = probs[0]
                    if random_number < prob_transition:
                        next_damage[i] = 0
                    else:
                        next_damage[i] = 1
                else:
                    random_number = np.random.random()
                    prob_transition = probs[1]
                    if random_number < prob_transition:
                        next_damage[i] = 1
                    else:
                        next_damage[i] = 0

        time_step += 1
        next_damage = np.array(next_damage).reshape(-1, 1)
        initial_states = np.array(initial_states)
        intermediate_array = change_state_damage(initial_states, next_damage)
        intermediate_array = intermediate_array.astype(int)
        new_states = apply_hfg_policy(intermediate_array, hfg)
        new_states = flatten(new_states)
        initial_states = list(initial_states)
        next_damage = change_to_opposite(new_states)
        damage = next_damage.copy()
        not_fixed = [i for i, v in enumerate(next_damage) if v == 1]
        calculate_state.append(new_states)

    rem_node = [0, 1, 2]
    calculate_state2 = {each: [] for each in range(len(calculate_state))}

    count = 0
    for each in calculate_state:
        for i in range(len(each)):
            if (each[i] == 0) and (i in action_rule):
                if i not in rem_node:
                    calculate_state2[count].append(i)
        count += 1

    #print("calculate_state2",calculate_state2)



    # Dictionaries to record values over time
    storage_values_over_time = {storage: [] for storage in water_storages}
    community_usage_over_time = {}  # Dictionary to store dict_community_usage over time
    initial_water_storages = water_storages.copy()

    # Calculate DNS and total load over time
    #total_dns_over_time = {}
    total_load_over_time = {}

    # Initialize dictionaries for each node
    for node in consumer_dicts.keys():
        #total_dns_over_time[node] = 0
        total_load_over_time[node] = 0





    # calculate average time of infrastructure failure
    time_failure = {}
    for key in calculate_state2:
        for nd in failed_indices:
            if nd in calculate_state2[key]:
                if nd not in time_failure:
                    time_failure[nd] = 0
                time_failure[nd] += 1

    for storage in water_storages.keys():
        storage_values_over_time[storage].append(initial_water_storages[storage])


    first_key = next(iter(calculate_state2))
    del calculate_state2[first_key]
    refill_quantity = 4500

    # Calculate total load and DNS over time
    for ts, failed_nodes in calculate_state2.items():
        # Calculate the water consumption for each storage
        dict_community_usage = {}
        for storage in water_storages.keys():
            #print(f"Storage {storage} at time step {time_step}: Initial water level {water_storages[storage]}")
            consumer_nodes_dependent_on_storage = set()
            #increment_applied = False

            # Check dependency for each failed node
            for ind in failed_indices:
                if ind in consumer_path_dict and storage in consumer_path_dict[ind]:
                    consumer_nodes_dependent_on_storage.update(consumer_path_dict[ind])
                    #print("consumer_path_dict_2", consumer_path_dict, ind,consumer_nodes_dependent_on_storage)

            # Filter consumer nodes that depend on the current storage
            consumer_nodes = [node for node in consumer_nodes_dependent_on_storage if node in consumer_dicts]
            #print("consumer_nodes", consumer_nodes)

            fixed = not(storage in calculate_state2.get(ts, []))


            if fixed:
                water_usage = sum(consumer_dicts[node] for node in consumer_nodes if node in consumer_dicts)
                # Adjust water storage level based on water usage
                if water_storages[storage] >= water_usage:
                    water_storages[storage] -= water_usage
                    water_storages[storage] += refill_quantity
                    # Ensure water storage doesn't exceed capacity
                    water_storages[storage] = min(water_storages[storage], initial_water_storages[storage])
                    for i in consumer_nodes:
                        dict_community_usage[i] = consumer_dicts[i]
                else:
                    for node in criticality:
                        if node in consumer_nodes:
                            water_usage = consumer_dicts[node]
                            if water_storages[storage] >= water_usage:
                                water_storages[storage] -= water_usage
                                dict_community_usage[node] = water_usage
                            else:
                                allocated_water = water_storages[storage]
                                water_storages[storage] = 0
                                dict_community_usage[node] = allocated_water
                                break
                    water_storages[storage] += refill_quantity
            else:
                water_usage = sum(consumer_dicts[node] for node in consumer_nodes if node in consumer_dicts)
                if water_storages[storage] >= water_usage:
                    water_storages[storage] -= water_usage
                    for i in consumer_nodes:
                        dict_community_usage[i] = consumer_dicts[i]
                else:
                    for node in criticality:
                        if node in consumer_nodes:
                            water_usage = consumer_dicts[node]
                            if water_storages[storage] >= water_usage:
                                water_storages[storage] -= water_usage
                                dict_community_usage[node] = water_usage
                            else:
                                allocated_water = water_storages[storage]
                                water_storages[storage] = 0
                                dict_community_usage[node] = allocated_water
                                break
                # Ensure water storage doesn't go negative
                water_storages[storage] = max(water_storages[storage], 0)

                #print(f"Time step {time_step}: No consumer nodes dependent on storage {storage}")

            storage_values_over_time[storage].append(water_storages[storage])


        # Save dict_community_usage for the current time step
        community_usage_over_time[ts] = dict_community_usage

    ts += 1 #Make up the timestep for initial storage in dictionary of storage


    return new_states, ts, calculate_state2, storage_values_over_time,time_failure,community_usage_over_time,consumer_path_dict

#Calculate_State2 {0: [34, 36], 1: [34, 35, 36, 37], 2: [34, 35, 36, 37, 40, 42, 44, 46, 48, 50, 52], 3: [34, 35, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], 4: [35, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], 5: [35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], 6: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], 7: [40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53], 8: [40, 41, 42, 43, 44, 45, 46, 47, 49, 51, 52, 53], 9: [40, 41, 42, 43, 44, 45, 46, 47, 49, 51, 53], 10: [41, 42, 43, 44, 45, 46, 47, 49, 51, 53], 11: [41, 43, 44, 45, 46, 47, 49, 51, 53], 12: [41, 43, 45, 46, 47, 49, 51, 53], 13: [41, 43, 45, 47, 49, 51, 53], 14: [41,

def simulate_damage_evolution_power(damage, initial_states, hfg, num_crew, action_rule, probs,
                              consumer_dicts, demand_list,consumer_path_dict,failed_indices):
    """

    :param damage:
    :param initial_states:
    :param hfg:
    :param num_crew:
    :param action_rule:
    :param probs:
    :param consumer_dicts:
    :param demand_list: This could be Demand not served in Power
    :return:
    """

    dns_dict = {}
    for key,each in zip(consumer_dicts.keys(), demand_list):
        dns_dict[key] = each

    time_step = 1
    next_damage = damage.copy()
    new_states = [0] * ((action_rule[-1]) + 1)

    calculate_state = []
    calculate_state.append(change_to_opposite(list(damage)))
    cycles = find_cycles(hfg)

    not_fixed = [i for i, v in enumerate(next_damage) if v == 1]

    while new_states != initial_states:
        # for i in not_fixed:
        #     damage_times[i] += 1

        current = get_current(action_rule, num_crew, next_damage)

        time_stamp = time_step

        for i in action_rule:
            if i in current:
                # repair_queue_times[i] += 1
                # if next_damage[i] == 1 and repair_times[i] is None:
                #     repair_times[i] = time_step  # Set repair start time if node is being repaired for the first time
                # Perform repair based on transition probabilities
                random_number = np.random.random()
                last_key = list(probs[2].keys())[-1]
                if time_step > last_key:
                    time_stamp = last_key
                prob_transition = probs[2].get(time_stamp)
                prob_transition = 1 - prob_transition
                if random_number >= prob_transition:
                    next_damage[i] = 0
                else:
                    next_damage[i] = 1
            else:
                if damage[i] == 0:
                    random_number = np.random.random()
                    prob_transition = probs[0]
                    if random_number < prob_transition:
                        next_damage[i] = 0
                    else:
                        next_damage[i] = 1
                else:
                    random_number = np.random.random()
                    prob_transition = probs[1]
                    if random_number < prob_transition:
                        next_damage[i] = 1
                    else:
                        next_damage[i] = 0

        time_step += 1
        next_damage = np.array(next_damage).reshape(-1, 1)
        initial_states = np.array(initial_states)
        intermediate_array = change_state_damage(initial_states, next_damage)
        intermediate_array = intermediate_array.astype(int)
        new_states = apply_hfg_policy(intermediate_array, hfg)
        new_states = flatten(new_states)
        initial_states = list(initial_states)
        next_damage = change_to_opposite(new_states)
        damage = next_damage.copy()
        not_fixed = [i for i, v in enumerate(next_damage) if v == 1]
        calculate_state.append(new_states)

    rem_node = [0, 1, 2]
    calculate_state2 = {each: [] for each in range(len(calculate_state))}
    count = 0
    for each in calculate_state:
        for i in range(len(each)):
            if (each[i] == 0) and (i in action_rule):
                if i not in rem_node:
                    calculate_state2[count].append(i)
        count += 1


    # Calculate DNS and total load over time
    total_dns_over_time = {}
    total_load_over_time = {}

    # Initialize dictionaries for each node
    for node in consumer_dicts.keys():
        total_dns_over_time[node] = 0
        total_load_over_time[node] = 0

    for failed_nodes in calculate_state2.values():
        for node in failed_nodes:
            if node in consumer_dicts:
                total_load_over_time[node] += consumer_dicts[node]
                if node in dns_dict:
                    total_dns_over_time[node] += dns_dict[node]

    dns_loss_vals = {}
    for i, v in total_dns_over_time.items():
        if v != 0 and total_load_over_time[i] != 0:
            val = v / total_load_over_time[i]
        else:
            val = 0
        dns_loss_vals[i] = val




    return new_states, time_step, calculate_state2, dns_loss_vals


def get_all_paths_index(paths_dict,graph):
    all_paths = []
    for each in paths_dict:
        result = unique_actions(paths_dict[each])
        all_paths.extend(result)

    all_paths_index = []
    for each in all_paths:
        all_paths_index.append(graph.nodes[each]['index'])
    #print(all_paths_index)
    return all_paths_index


# Function to get node name by index
def get_node_name_by_index(graph, index):
    for node in graph.nodes():
        if graph.nodes[node].get('index') == index:
            return node


def find_paths_to_failed_nodes(graph, source_nodes, destination_nodes):
    source_names = []
    for i in source_nodes:
        source_names.append(get_node_name_by_index(graph, i))

    destination_names = []
    for i in destination_nodes:
        destination_names.append(get_node_name_by_index(graph, i))

    # Find paths from source nodes to destination nodes
    paths = {}
    for source in source_names:
        paths[source] = {}
        for dest in destination_names:
            if nx.has_path(graph, source, dest):
                paths[source][dest] = list(nx.all_simple_paths(graph, source=source, target=dest))

    dicts_res = {}
    for i in paths:
        for j in paths[i]:
            dicts_res[graph.nodes[j]['index']] = paths[i][j]
    return dicts_res


def execute_simulation(infra, alpha, graph, predetermined_steps, lambda_parameter, num_crew, failed_nodes,dns_dict,ratio=[0.4,0.6]):
    """

    """
    # Find indices corresponding to failed nodes based on the specified node names


    failed_indices = []
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        #print("first",idx,node,data)
        for failed_node in failed_nodes:
            #print("node, failed_node", node.split()[-1][1:-1], failed_node)
            if failed_node== node.split()[-1][1:-1]:
                #print("second",idx,node,data)
                failed_indices.append(idx)

    # Find paths from source nodes to destination nodes with the same type and 'community' attribute set to True
    paths_dict = find_paths_with_same_type_and_community(graph, failed_indices)
    #print("paths_dict",paths_dict)


    consumer_list = []
    consumer_list_index = []
    consumer_path_dict = {}
    for each in paths_dict.keys():
        consumer_path_dict[each] = set()
        for cons in paths_dict[each]:
            index = graph.nodes[cons[-1]]['index']
            all_indices = [graph.nodes[node]['index'] for node in cons]
            consumer_list_index.append(index)
            consumer_list.append(cons[-1])
            consumer_path_dict[each].update(all_indices)

    hfg, ind = get_array(graph)


    # Getting the unique actions
    all_paths_index = get_all_paths_index(paths_dict,graph)


    paths_to_failed =  find_paths_to_failed_nodes(graph, [0], failed_indices)
    paths_to_failed_index = list(set(get_all_paths_index(paths_to_failed,graph)))


    correct_paths = list(set(paths_to_failed_index + all_paths_index))


    last_node = correct_paths[-1] + 1
    hfg_new = get_new_hfg(hfg, correct_paths, last_node)

    # This is the initial state of the functionalities
    num_rows, num_cols = 1, correct_paths[-1] + 1
    initial_state = np.ones((num_rows, num_cols))
    initial_state = initial_state.astype(int)
    initial_state = list(initial_state[0])

    # Get index of failed nodes
    initial_damage_array = np.zeros((1, correct_paths[-1] + 1))[0]
    for i in failed_indices:
        initial_damage_array[i] = 1

    # predetermined_steps = 20
    transition_matrix = get_transition_matrix(predetermined_steps, lambda_parameter)

    if alpha == 'sc':
        action = get_action_rule_metrics_weighted(graph, "generates 'power' at 'Generator1'", "SVS", "Criticality",
                                              ratio,alpha,consumer_path_dict)

    elif alpha == 's':
        action = get_action_rule_metrics(graph, "generates 'power' at 'Generator1'", "SVS",alpha,\
                                         consumer_path_dict)

    elif alpha == 'c':
        action = get_action_rule_metrics(graph, "generates 'power' at 'Generator1'", "Criticality",alpha,\
                                         consumer_path_dict)

    else:
        action = get_action_rule(hfg_new)

    action_rule = get_power_action(action, correct_paths)

    total_demand_dicts = {}
    for i, z in zip(consumer_list, consumer_list_index):
        if infra == 'power':
            total_demand_dicts[z] = graph.nodes[i]['Power_Total_Load']
        elif infra == 'water':
            total_demand_dicts[z] = graph.nodes[i]['Water_Total_Load']
        else:
            pass


    water_storages_dict = {}
    get_storage = []
    for nd in consumer_path_dict.values():
        get_storage.extend(nd)



    for each_nd in graph.nodes(data=True):
        if "WaterStorageTank" in each_nd[1]['resource']:
            ind = each_nd[1]['index']
            if ind in get_storage:
                water_storages_dict[each_nd[1]['index']] = each_nd[1]['Storage_Capacity']

    #criticality
    all_nodes = []
    for i in consumer_path_dict.values():
        all_nodes.extend(i)
    criticality = {}
    for nd in graph.nodes(data=True):
        ind = nd[1]['index']
        if ind in all_nodes:
            crit = nd[1]['Criticality']
            criticality[ind] = crit

    criticality =  dict(sorted(criticality.items(), key=lambda item: item[1],reverse=True))







    if infra == "power":

        new_states, time_step, calculate_state2, dns_loss_vals = simulate_damage_evolution_power(initial_damage_array,
                                                                                       initial_state, hfg_new, num_crew,
                                                                                       action_rule, transition_matrix,
                                                                                       total_demand_dicts, dns_dict,consumer_path_dict,failed_indices)
        return new_states, time_step, failed_indices, calculate_state2, dns_loss_vals, consumer_list_index

    elif infra == "water":

        new_states, time_step, calculate_state2, storage_values_over_time,time_failure,community_usage_over_time,consumer_path_dict = simulate_damage_evolution_water(initial_damage_array,
                                                                                                 initial_state, hfg_new,
                                                                                                 num_crew,
                                                                                                 action_rule,
                                                                                                 transition_matrix,
                                                                                                 total_demand_dicts,
                                                                                                 consumer_path_dict,failed_indices,water_storages_dict,criticality)
        return new_states, time_step, failed_indices, calculate_state2, storage_values_over_time,time_failure,community_usage_over_time, consumer_list_index,consumer_path_dict

    else:
        return "Simulation Evolution Fails!!! Choose correct Infrastructure!"





