import networkx as nx
from simulation import get_dependent_nodes
from simulation import execute_simulation
import argparse
from impact_analysis import *
from collections import defaultdict


# Results
def calc_cumm_over_time(calculate_state_list):
    new_ls = []
    for ls in calculate_state_list:
        new_dict = {}
        for k, v in ls.items():
            new_dict[k] = len(v)
        new_ls.append(new_dict)

    ct = 0
    keys_to_delete = {}
    for each in new_ls:
        keys_to_delete[ct] = []
        for k2, v2 in each.items():
            if v2 == 0:
                keys_to_delete[ct].append(k2)
        ct += 1

    # Delete keys from dictionaries in new_ls
    for idx, keys in keys_to_delete.items():
        if len(keys) > 0:
            for key in keys:
                if key in new_ls[idx]:
                    del new_ls[idx][key]

    # Initialize a dictionary to store the sums of values for each key
    sums = {}

    # Count the number of dictionaries
    count = len(new_ls)

    # Iterate over each dictionary in the list
    for d in new_ls:
        # Iterate over each key-value pair in the dictionary
        for key, value in d.items():
            # Update the sum for the key
            sums[key] = sums.get(key, 0) + value

    # Calculate averages
    averages = {key: value / count for key, value in sums.items()}

    return averages


def time_to_repair(calculate_state):
    time_to_repair_failed = {}
    for key, value in calculate_state.items():

        for each in value:
            if each not in time_to_repair_failed.keys():
                time_to_repair_failed[each] = 1
            else:
                time_to_repair_failed[each] += 1
    return time_to_repair_failed


# Cascading Effect Plot- In each time step how many of it dependents became non-functional
def cascading_effect(graph,calculate_state, failed_index): #TODO: remove the community and generalize
    dep_dict = {}
    for i in failed_index:
        if i == 3:
            community = 'Community_1'
        elif i == 5:
            community = 'Community_2'
        else:
            community = 'Community_3'
        dep = get_dependent_nodes(graph, i, community)
        if i not in dep_dict:
            dep_dict[i] = {}  # Initialize inner dictionary for i if not present
        for k, v in calculate_state.items():
            if k not in dep_dict[i]:
                dep_dict[i][k] = []  # Initialize list for k if not present
            for each in v:
                if each in dep:
                    dep_dict[i][k].append(each)  # Append to the list

    node_counts = {}
    for key, value in dep_dict.items():
        node_counts[key] = {}
        for time, nodes in value.items():
            node_counts[key][time] = len(nodes)

    # Drop keys from node_counts dictionary whose values are 0
    keys_to_delete = []

    for key, value in node_counts.items():
        for k2, v2 in value.items():
            if v2 == 0:
                keys_to_delete.append((key, k2))

    # Delete keys from node_counts dictionary
    if len(keys_to_delete) > 0:
        for key, k2 in keys_to_delete:
            del node_counts[key][k2]

    keys_to_delete2 = []

    for key, value in node_counts.items():
        if len(value) == 0:
            keys_to_delete2.append(key)

    if len(keys_to_delete2) > 0:
        for key in keys_to_delete2:
            del node_counts[key]

    return dep_dict, node_counts


def cal_average_dependence(node_count_list):
    result = {}

    for d in node_count_list:
        for key, sub_dict in d.items():
            if key not in result:
                result[key] = {}
            for sub_key, value in sub_dict.items():
                if sub_key not in result[key]:
                    result[key][sub_key] = 0
                result[key][sub_key] += value

    # Calculate average
    for key, sub_dict in result.items():
        for sub_key, value in sub_dict.items():
            result[key][sub_key] = value / len(node_count_list)
    return result


def calc_average_time(data):
    # Initialize a dictionary to store the sum of values for each key
    sum_values = {}
    # Initialize a dictionary to store the count of dictionaries containing each key
    count_values = {}

    # Iterate over the list of dictionaries
    for d in data:
        # Iterate over each inner dictionary
        for key, value in d.items():
            # Update sum and count for each key
            sum_values[key] = sum_values.get(key, 0) + value
            count_values[key] = count_values.get(key, 0) + 1

    # Initialize a dictionary to store the average of values for each key
    average_values = {}

    # Calculate the average for each key
    for key, total in sum_values.items():
        count = count_values[key]
        average_values[key] = total / count
    return average_values

# Function to combine dictionaries by key and sum values
def combine_dicts(list_of_dicts):
    combined_dict = defaultdict(dict)
    for d in list_of_dicts:
        for key in d:
            for k, v in d[key].items():
                combined_dict[key][k] = combined_dict[key].get(k, 0) + v
    return dict(combined_dict)

# Function to calculate average values
def calculate_average(combined_dict, num_dicts):
    average_dict = {}
    for key in combined_dict:
        average_dict[key] = {}
        for k, v in combined_dict[key].items():
            average_dict[key][k] = v / num_dicts
    return average_dict


def sum_values(average_dict):
    key_sums = {}
    for key in average_dict:
        for k, v in average_dict[key].items():
            if k not in key_sums:
                key_sums[k] = 0
            key_sums[k] += v
    return key_sums

def monte_carlo_power( infra, alpha, graph, predetermined_steps, lambda_parameter, num_crew, failed_nodes, dns_dict, num_simulations=20,ratio=[0.4,0.6]):
    # Define parameters
    total_time_step = {}
    time_to_repair_failed_list = []
    node_counts_list = []
    calculate_state_list = []
    cumm_dns_load = []

    # Execute the simulation multiple times

    for sim in range(num_simulations):
        new_states, time_step, failed_index, calculate_state, dns_loss_vals,consumer_list_index = execute_simulation(
            infra, alpha, graph,predetermined_steps, lambda_parameter,
            num_crew, failed_nodes,dns_dict,ratio)

        total_time_step[sim] = time_step

        time_to_repair_failed = time_to_repair(calculate_state)
        time_to_repair_failed_list.append(time_to_repair_failed)
        average_time_to_repair_failed = calc_average_time(time_to_repair_failed_list)

        dep_dict, node_counts = cascading_effect(graph,calculate_state, failed_index)
        node_counts_list.append(node_counts)
        cascade_result = cal_average_dependence(node_counts_list)

        calculate_state_list.append(calculate_state)
        average_calc_cumm_over_time = calc_cumm_over_time(calculate_state_list)

        cumm_dns_load.append(dns_loss_vals)
    return total_time_step, average_time_to_repair_failed, cascade_result, average_calc_cumm_over_time, cumm_dns_load, consumer_list_index

def monte_carlo_water(infra, alpha, graph, predetermined_steps, lambda_parameter, num_crew, failed_nodes,
                      dns_dict, num_simulations=20, ratio=[0.4, 0.6]):
    # Define parameters
    storage_values_over_time_list = []
    time_failure_list = []
    community_usage_list = []
    time_step_list =[]




    # Execute the simulation multiple times
    for sim in range(num_simulations):



        new_states, time_step, failed_indices, calculate_state2,\
        storage_values_over_time, time_failure, community_usage_over_time, consumer_list_index,consumer_path_dict = execute_simulation(
            infra, alpha, graph,predetermined_steps, lambda_parameter,num_crew, failed_nodes,dns_dict,ratio)

        storage_values_over_time_list.append(storage_values_over_time)
        time_failure_list.append(time_failure)
        community_usage_list.append(community_usage_over_time)
        time_step_list.append(time_step)


        # Combine all dictionaries
        community_usage_dict = combine_dicts(community_usage_list)



    max_time = max(time_step_list)
    for store_val in storage_values_over_time_list:
        for each in store_val:

            begin_node = store_val[each][0]
            len_store = len(store_val[each])
            store_to_add = max_time - len_store
            store_val[each] = store_val[each] + ([begin_node] * store_to_add)



    # Average Storage over time -storage_values_over_time_list
    # Initialize a dictionary to store the sum of values for each storage
    sum_storage_values = {}

    # Find the maximum length of any list
    max_length = max(len(values) for iteration in storage_values_over_time_list for values in iteration.values())

    # Pad all lists to the maximum length with zeros
    for storage_values_over_time in storage_values_over_time_list:
        for storage, values in storage_values_over_time.items():
            if len(values) < max_length:
                storage_values_over_time[storage] += [0] * (max_length - len(values))

    # Aggregate values from all iterations
    for storage_values_over_time in storage_values_over_time_list:
        for storage, values in storage_values_over_time.items():
            if storage not in sum_storage_values:
                sum_storage_values[storage] = [0] * len(values)
            for i in range(len(values)):
                sum_storage_values[storage][i] += values[i]

    # Calculate the average values for each storage
    average_storage_values = {storage: [value / num_simulations for value in values] for storage, values in
                              sum_storage_values.items()}


    # Average Time Failure - time_failure
    # Initialize a dictionary to store the sum of values for each key
    sum_time_failure = {}

    # Aggregate values from all iterations
    for time_s in time_failure_list:
        for key, value in time_s.items():
            if key not in sum_time_failure:
                sum_time_failure[key] = 0
            sum_time_failure[key] += value

    # Calculate the average values for each key
    average_time_failure = {key: value / num_simulations for key, value in sum_time_failure.items()}



    #Average Community Usage Over time -community_usage_over_time

    # Calculate average
    num_dicts = len(community_usage_list)
    avg_community_usage = calculate_average(community_usage_dict, num_dicts)

    # Calculate sum of values for each key
    avg_consumer_key_sums = sum_values(avg_community_usage)

    f_time_step = max(time_step_list)


    return new_states, f_time_step, failed_indices, calculate_state2,\
            average_storage_values , average_time_failure, avg_consumer_key_sums, consumer_list_index,consumer_path_dict







def main():
    parser = argparse.ArgumentParser(
        description='Simulation script for failure and repair processes with impact analysis')
    parser.add_argument('--graph', type=str, help='Path to the graph file', required=True)
    parser.add_argument('--infra', type=str, help='infra', required=True)
    parser.add_argument('--alpha', type=str, help='Alpha value (a,s, c, sc)', required=True, choices=['a','s', 'c', 'sc'])
    parser.add_argument('--time-steps', type=int, help='Number of time steps',required=True)
    parser.add_argument('--mean', type=float, help='Mean value',required=True)
    parser.add_argument('--num-crews', type=int, help='Number of repair crews',required=True)
    parser.add_argument('--failed-nodes', nargs='+', help='List of failed nodes', required=True)
    parser.add_argument('--dns-dicts', help='List of demand not served values for each area', nargs='+',required=False, type=int)
    parser.add_argument('--sims', type=int, help='Number of simulations',default=20)
    parser.add_argument('--ratio', type=float, nargs='+', help='Ratio parameter for alpha "sc"',default=[0.4, 0.6])


    args = parser.parse_args()

    # # Check if all required arguments are provided
    # if not all(vars(args).values()):
    #     parser.error('All arguments are required')

    # Run the simulation with provided arguments
    graph_path = args.graph
    infra = args.infra
    alpha = args.alpha
    time_steps = args.time_steps
    mean = args.mean
    num_crew = args.num_crews
    failed_nodes = args.failed_nodes
    dns_dict = args.dns_dicts
    sims = args.sims
    ratio = args.ratio

    #
    # # Adjusting based on alpha value
    # if args.alpha == 'sc':
    #     parser.add_argument('--ratio', nargs=2, type=float, help='Ratio parameter for alpha "sc"', required=True)
    #     args = parser.parse_args()  # Re-parse arguments to include ratio
    # else:
    #     if args.ratio is not None:
    #         parser.error("Ratio parameter should not be provided for alpha 'a', 's', or 'c'.")

    # Load the graph
    G_hfg = nx.read_graphml(graph_path)
    # print(G_hfg)
    if infra == "power":
        total_time_step, average_time_to_repair_failed, cascade_result, average_calc_cumm_over_time, \
        cumm_dns_load,consumer_list_index = monte_carlo_power(infra,alpha, G_hfg , time_steps, mean, num_crew, \
                                                        failed_nodes,dns_dict, sims, ratio)
    elif infra == "water":
        new_states, time_step, failed_indices, calculate_state2, \
            storage_values_over_time, time_failure, community_usage_over_time, consumer_list_index,consumer_path_dict= monte_carlo_water(infra,alpha, G_hfg , time_steps, mean, num_crew, \
                                                        failed_nodes,dns_dict, sims, ratio)
        total_load = {}
        for i in  community_usage_over_time:
            for nd in G_hfg.nodes(data=True):
                if i == nd[1]["index"]:
                    total_load[i] = nd[1]["Water_Total_Load"] * (time_step)
        #print("community_usage_over_time",community_usage_over_time)
        #print("total_load",total_load,time_step)

        ratio_usage = {}
        for i in community_usage_over_time:
            ratio_usage[i]= (community_usage_over_time[i])/total_load[i]




    else:
        "Failed!!"



    failed_indices_dict = {}
    for idx, (node, data) in enumerate(G_hfg.nodes(data=True)):
        # print("first",idx,node,data)
        for failed_node in failed_nodes:
            # print("node, failed_node", node.split()[-1][1:-1], failed_node)
            if failed_node == node.split()[-1][1:-1]:
                # print("second",idx,node,data)
                failed_indices_dict[idx] = failed_node


    if alpha == 'sc':
        order = "SVS and Criticality"
    elif alpha == 's':
        order = "SVS"
    elif alpha == 'c':
        order = "Criticality"
    elif alpha == 'a':
        order = "No Preference"

    #Visualizations
    if infra == "power":
        #average_repair_plots(average_time_to_repair_failed,failed_nodes,list(failed_indices_dict.keys()), sims,order)
        #print("failed_indices_dict",failed_indices_dict)


        average_number_functionalities_over_time(cascade_result,failed_indices_dict,order)

        average_time_outages_by_location_before_repair(average_time_to_repair_failed, G_hfg,consumer_list_index,order)

        proportion_dns_over_load(cumm_dns_load, G_hfg , consumer_list_index, order)
        #pass

    elif infra == "water":
        water_storage_available_over_time(G_hfg,storage_values_over_time,order)
        water_infratructure_failure_over_time(G_hfg, time_failure, order,sims)
        get_community_usage_over_time(G_hfg,ratio_usage,consumer_list_index,consumer_path_dict,order)
        #pass


    # Print or save the results
    print("Simulation completed successfully!")


if __name__ == '__main__':
    main()

# python main.py --graph "data/community_hfg_model.graphml" --infra "power" --alpha "c" --time-steps 15 --mean 0.3 --num-crews 2 --failed-nodes 'Powerline2' 'Powerline7' 'Powerline12' --dns-dicts 100 200 500 1000 200 100 500 200 100 1000 --sims 100 --ratio 0.4 0.6
#python main.py --graph "data/community_hfg_model.graphml" --infra "water" --alpha "sc" --time-steps 15 --mean 0.3 --num-crews 2 --failed-nodes 'WaterPipeline1' 'WaterPipeline12' 'WaterPipeline13'  --dns-dicts 100 200 500 1000 200 100 500 200 100 1000 --ratio 0.8 0.2



#For power infrastructure we had plots of Demand not Served/Total Load across communities. What interesting plot can we have for Water infrastructure to show changes in water usage over