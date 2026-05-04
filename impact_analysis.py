import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves figures to files without showing windows
import matplotlib.pyplot as plt
import numpy as np



def average_repair_plots(average_time_to_repair_failed,failed_nodes,failed_indicies, sim_steps,order):
    """
    Plots Average Repair Time for Functionalities that are Affected
    :param average_time_to_repair_failed:
    :return:
    """


    #Plot of Dynamics
    # Data
    sliced_dict = {k: average_time_to_repair_failed[k] for k in list(average_time_to_repair_failed.keys()) if k in failed_indicies}
    #print("average_time_to_repair_failed",average_time_to_repair_failed)
    ##print("sliced_dict",sliced_dict)

    # Extract keys and values from the data dictionary
    values = list(sliced_dict.values())

    # Plotting
    plt.figure(figsize=(10, 6))  # Adjust figure size
    #print("failed_nodes, values",failed_nodes, values)
    plt.bar(failed_nodes, values, color='darkorange', edgecolor='black')  # Add edge color for better contrast
    plt.xlabel('Functionality', fontsize=10)  # Increase font size for axis labels
    plt.ylabel('Average Time to Repair', fontsize=10)
    #plt.title(f'Average Repair Time for Functionality for {sim_steps} Simulations', fontsize=10, fontweight='bold')  # Enhance title
    plt.xticks(failed_nodes, fontsize=10)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=12)  # Increase font size for y-axis ticks
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f'power_average_repair_time_{order}.png', dpi=300, bbox_inches='tight')  # Save as high-resolution image






def average_number_functionalities_over_time(cascade_result, failed_indices_dict, order):
    """
    Plot of Average Number of Functionalities Affected Over Time (By Metric)
    :param cascade_result:
    :return:
    """
    fontsize = 20
    plt.figure(figsize=(12, 8))

    # Collect all time values to determine max timestep
    all_times = set()
    for time_nodes in cascade_result.values():
        all_times.update(time_nodes.keys())
    max_time = max(all_times)

    # Plot each failure scenario
    for key, time_nodes in cascade_result.items():
        times = list(time_nodes.keys())
        node_counts_values = list(time_nodes.values())
        plt.plot(times, node_counts_values, label=f'{failed_indices_dict[key]}')

    plt.xlabel('Simulation Timestep', fontsize=fontsize)
    plt.ylabel('Functionality Loss', fontsize=fontsize)

    # Set integer ticks using plt.xticks
    plt.xticks(range(0, max_time + 1), fontsize=18)  # Integer ticks only

    plt.yticks([0, 2, 4, 6, 8, 10], fontsize=18)  # Optional: tune for your data
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'power_average_functionalities_over_time_{order}.png', dpi=300, bbox_inches='tight')


def average_time_outages_by_location_before_repair(average_time_to_repair_failed,graph, community,order):
    """
    Plot of Average Time of Outages by Location Before Repair(By Metrics) by Grouping data by community
    :param average_time_to_repair_failed:
    :return:
    """
    fontsize = 20
    #print("average_time_to_repair",average_time_to_repair_failed)


    comm_dict = {}
    for i in community:
        if i in average_time_to_repair_failed.keys():
            comm_dict[i] = average_time_to_repair_failed[i]

    #print("comm_dict",comm_dict)

    comm_indx_dict = {}
    all_comms_dict = {}
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if idx in community:
            comm_indx_dict[idx] = node
            cc = list(data.keys())[1]  # Assuming 'data' has enough entries and index 1 exists.
            if cc not in all_comms_dict:
                all_comms_dict[cc] = []  # Initialize with an empty list if key not found
            all_comms_dict[cc].append(idx)  # Append the index to the list of the corresponding community category




    combined_dict = {}
    for community, indices in all_comms_dict.items():
        # Initialize each community with an empty dictionary
        community_dict = {}
        for idx in indices:
            # Map each index to its corresponding description in comm_indx_dict
            community_dict[idx] = comm_indx_dict[idx]
        # Assign the filled community dictionary to the combined dictionary
        combined_dict[community] = community_dict


    # Map keys in comm_dict to the corresponding last string in comm1, comm2, and comm3

    mapped_keys = {}
    for community_name, community_dict in combined_dict.items():
        for key, value in community_dict.items():
            # Extract the location by splitting the string and removing the single quotes
            mapped_keys[key] = value.split(" at ")[-1].strip("'")


    # Sample data for plotting
    width = 1  # Adjusted width for better visualization
    groupgap = 1

    # Dictionary to store the y-values for each community
    y_values = {}

    # Loop through each community and calculate sorted key values
    for comm_name, comm_val in combined_dict.items():
        y_values[comm_name] = [comm_dict.get(key, np.nan) for key in sorted(comm_val.keys())]

    # Colors for each community (can use a cycle from itertools if more communities)
    colors = ['r', 'b', 'g', 'y', 'm', 'c']
    # Dictionary to store rects for each community
    rects_dict = {}

    # Initialize the offset
    offset = 0
    ind = []
    fig, ax = plt.subplots(figsize=(12, 9))  # Increased figure size for poster
    # Calculate and plot bars for each community
    for (comm_key, y), color in zip(y_values.items(), colors):
        # Generate x values starting from current offset
        x = np.arange(len(y)) + offset
        ind.extend(x)
        rects = ax.bar(x, y, width, color=color, edgecolor="black", label=comm_key)
        rects_dict[comm_key] = rects  # Store the rects for possible future use

        # Update offset for next community
        offset += len(y) + groupgap


    # Configure plot aesthetics
    ax.set_ylabel('Average Outage Time Before Repair', fontsize=fontsize)
    ax.set_xlabel('',fontsize=fontsize)  # Adjust fontsize as needed
    #ax.set_title(f'Average Time of Outages by Location Before Repair(Total Repair Order by {order})', fontsize=fontsize)
    ax.set_xticks(ind)

    sorted_keys = sorted(key for comm in combined_dict.values() for key in comm.keys())
    labels = [mapped_keys[key] for key in sorted_keys]
    ax.set_yticks([0, 2, 4, 6, 8,10,12,14]) #TODO: remove later
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'power_average_outage_times_{order}.png', dpi=300, bbox_inches='tight')  # Save as high-resolution image



def proportion_dns_over_load(cumm_dns_load,graph,community,order):
    """
    Plot of Proportion of DNS over Total Load(Total Repair Order by Criticality)
    :return:
    """

    svs_dict = {}
    for each, vals in graph.nodes(data=True):
        for x in community:
            if x == vals['index']:
                svs_dict[x] = vals['SVS']
    #print("community", cumm_dns_load, community,svs_dict)
    # Initialize a dictionary to hold the sum of values and the count of non-zero values for each key
    fontsize = 20
    sum_values = {key: 0 for key in cumm_dns_load[0].keys()}

    # Iterate over the list of dictionaries and sum up the values for each key
    for d in cumm_dns_load:
        for key, value in d.items():
            sum_values[key] += value

    # Calculate the average for each key
    num_dicts = len(cumm_dns_load)
    cumm_dns_load_avg = {key: (sum_value / num_dicts) * 100 for key, sum_value in sum_values.items()}

    comm_indx_dict = {}
    all_comms_dict = {}
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if idx in community:
            comm_indx_dict[idx] = node
            cc = list(data.keys())[1]  # Assuming 'data' has enough entries and index 1 exists.
            if cc not in all_comms_dict:
                all_comms_dict[cc] = []  # Initialize with an empty list if key not found
            all_comms_dict[cc].append(idx)  # Append the index to the list of the corresponding community category

    combined_dict = {}
    for community, indices in all_comms_dict.items():
        # Initialize each community with an empty dictionary
        community_dict = {}
        for idx in indices:
            # Map each index to its corresponding description in comm_indx_dict
            community_dict[idx] = comm_indx_dict[idx]
        # Assign the filled community dictionary to the combined dictionary
        combined_dict[community] = community_dict



    com_cumm_vals ={key: 0 for key in all_comms_dict.keys()} #{"Community1": 0, "Community2": 0, "Community3": 0}


    for comm, nodes in all_comms_dict.items():
        sum_vals = sum(cumm_dns_load_avg[node] for node in nodes if node in cumm_dns_load_avg)
        avg_val = sum_vals / len(nodes)
        com_cumm_vals[comm] = avg_val



    svs_comm_dict = {}
    for i in all_comms_dict.keys():
        for j in svs_dict.keys():
            if j in all_comms_dict[i]:
                svs_comm_dict[i] = svs_dict[j]



    # Colors for each community (can use a cycle from itertools if more communities)
    colors = ['r', 'b', 'g', 'y', 'm', 'c']

    plt.figure(figsize=(12, 8))

    nodes = list(com_cumm_vals .keys())
    total_usage = list(com_cumm_vals.values())


    bars = plt.bar(nodes, total_usage, color=colors)

    for i, bar in enumerate(bars):
        additional_val = list(svs_comm_dict.values())[i]
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'SVS={additional_val}', ha='center', va='bottom',
                 fontsize=14)

    #plt.xlabel('Communities')
    plt.yticks([0, 2, 4, 6, 8, 10,12,14,16,18,20], fontsize=18)  # TODO: Remove later
    # plt.xticks(fontsize=18)
    plt.xlabel('',fontsize=20)  # Adjust fontsize as needed

    plt.ylabel('DNS/Total Load(%)', fontsize=fontsize)
    plt.savefig(f'power_dns_{order}.png', dpi=300, bbox_inches='tight')  # Save as high-resolution image




#water
#community_usage_over_time
def water_storage_available_over_time(graph, storage_values_over_time,order):
    new_storage ={}
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if idx in storage_values_over_time:
            new_storage[data['resource']] = storage_values_over_time[idx]


        #print(idx, node,data)

    # Plot storage values over time
    plt.figure(figsize=(10, 6))

    for storage, values in new_storage.items():
        plt.plot(values, label=f'{storage}')


    plt.xticks(fontsize=18)  # Adjust fontsize of x-axis ticks as needed
    plt.xlabel('Time Step',fontsize=20)
    plt.ylabel('Water Level (litres)')
    #plt.title(f'Average Water Storage Levels Over Time(Total Repair Order by {order})')
    plt.legend()
    plt.savefig(f'average_water_storage_level_{order}.png', dpi=300, bbox_inches='tight')  # Save as high-resolution image


def water_infratructure_failure_over_time(graph, time_failure,order,sims):
    #print("time_failure",time_failure)
    new_storage = {}
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if idx in time_failure:
            new_storage[data['resource']] = time_failure[idx]

    colors = ['r', 'b', 'g', 'y', 'm', 'c']
    # Plot time failure
    plt.figure(figsize=(10, 6))

    plt.bar(new_storage.keys(), new_storage.values(), color=colors)

    #plt.xlabel('Node')
    plt.xlabel('',fontsize=20)  # Adjust fontsize as needed
    plt.xticks(fontsize=18)  # Adjust fontsize of x-axis ticks as needed
    plt.ylabel(f'Average Time to Repair')
    #plt.title(f'Average Time to Repair for each Node in {sims} simulations(order by {order})')
    plt.savefig(f'average_repair_time_{order}.png', dpi=300, bbox_inches='tight')  # Save as high-resolution image


def get_community_usage_over_time(graph,total_usage_per_node,community,consumer_path_dict,order):
    #print("total_usage_per_node",total_usage_per_node)

    rep_ind = {}
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if idx in total_usage_per_node:
            rep_ind[idx] = data['resource']
    #print("rep_ind",rep_ind)
    #print(community)

    svs_dict = {}
    for each, vals in graph.nodes(data=True):
        for x in community:
            if x == vals['index']:
                svs_dict[x] = vals['SVS']
    #print("svs_dict", svs_dict)

    comm_indx_dict = {}
    all_comms_dict = {}

    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if idx in community:
            comm_indx_dict[idx] = node
            cc = list(data.keys())[1]  # Assuming 'data' has enough entries and index 1 exists.
            if cc not in all_comms_dict:
                all_comms_dict[cc] = []  # Initialize with an empty list if key not found
            all_comms_dict[cc].append(idx)


    #for each in graph.nodes(data=True):


    colors = ['r', 'b', 'g', 'y', 'm', 'c']
    # Calculate the sum of water availability ratios for each node

    new_total_usage_per_node= {}
    svs_dict_node ={}
    for i in total_usage_per_node:
        new_total_usage_per_node[rep_ind[i]] = total_usage_per_node[i]
        svs_dict_node[rep_ind[i]] = svs_dict[i]



    # Create a mapping from pipelines to communities
    pipeline_to_community = {}
    for community, nodes in all_comms_dict.items():
        for node in nodes:
            for key, value in consumer_path_dict.items():
                if node in value:
                    pipeline_to_community[rep_ind[node]] = community


    # Replace pipeline identifiers with community names in new_total_usage_per_node
    new_total_usage_per_community = {}

    for pipeline, usage in new_total_usage_per_node.items():
        community = pipeline_to_community.get(pipeline, pipeline)  # Default to pipeline if no match found
        if community in new_total_usage_per_community:
            new_total_usage_per_community[community] += usage
        else:
            new_total_usage_per_community[community] = usage


    svs_community = {}
    for xx in svs_dict_node:
        if xx in pipeline_to_community:
            svs_community[pipeline_to_community[xx]] = svs_dict_node[xx]





    # Plot total water availability for each node
    plt.figure(figsize=(12, 8))

    nodes = list(new_total_usage_per_community.keys())
    total_usage = list(new_total_usage_per_community.values())


    bars = plt.bar(nodes, total_usage, color=colors)

    for i, bar in enumerate(bars):
        additional_val = list(svs_community.values())[i]

        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() /2, yval, f'SVS={additional_val}', ha='center', va='bottom',
                 fontsize=14)

    #plt.xlabel('Communities')
    plt.xlabel('',fontsize=20)  # Adjust fontsize as needed
    plt.xticks(fontsize=18)
    plt.ylabel('Total Water Available/Total Demand')
    #plt.title(f'Ratio of Total Water Available over Total Demand for Each Community(Order by {order})')
    plt.savefig(f'average_water_availiabilty_{order}.png', dpi=300, bbox_inches='tight')  # Save as high-resolution image


def plot_water_availability_grouped(results_by_alpha, graph, community, consumer_path_dict,
                                    primary_alpha=None):
    """
    Produces Figure 14 from the paper:
    Grouped bar chart of Total Water Available / Total Demand per community,
    with one bar group per community and one bar per repair ordering metric.

    :param results_by_alpha: dict of {alpha_label: total_usage_per_node}
        e.g. {"Criticality": {...}, "SVS": {...}, "SVS and Criticality": {...}}
        Each value is the ratio_usage dict from main.py (node_index -> ratio value)
    :param graph: NetworkX graph
    :param community: list of community consumer node indices
    :param consumer_path_dict: dict mapping failed node indices to downstream path indices
    :param primary_alpha: optional label of the primary alpha (used in filename so
        each run saves a separately named file for comparison e.g.
        water_availability_grouped_Criticality.png)
    """
    import numpy as np

    fontsize = 20

    # Build rep_ind: node_index -> resource_name
    rep_ind = {}
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if idx in community:
            rep_ind[idx] = data['resource']

    # Build SVS lookup
    svs_dict = {}
    for each, vals in graph.nodes(data=True):
        for x in community:
            if x == vals['index']:
                svs_dict[x] = vals['SVS']

    # Build all_comms_dict: community_name -> [node_indices]
    all_comms_dict = {}
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if idx in community:
            cc = list(data.keys())[1]
            if cc not in all_comms_dict:
                all_comms_dict[cc] = []
            all_comms_dict[cc].append(idx)

    # Build pipeline_to_community mapping: resource_name -> community_name
    pipeline_to_community = {}
    for comm_name, nodes in all_comms_dict.items():
        for node in nodes:
            for key, value in consumer_path_dict.items():
                if node in value:
                    pipeline_to_community[rep_ind[node]] = comm_name

    # For each alpha, aggregate ratios by community
    # community_results: {alpha_label: {community_name: aggregated_ratio}}
    community_results = {}
    for alpha_label, total_usage_per_node in results_by_alpha.items():
        per_community = {}
        for node_idx, ratio_val in total_usage_per_node.items():
            if node_idx not in rep_ind:
                continue
            resource = rep_ind[node_idx]
            comm = pipeline_to_community.get(resource)
            if comm is None:
                continue
            if comm in per_community:
                per_community[comm] += ratio_val
            else:
                per_community[comm] = ratio_val
        community_results[alpha_label] = per_community

    # Get consistent community order
    all_communities = sorted(set(
        c for pc in community_results.values() for c in pc.keys()
    ))
    alpha_labels = list(results_by_alpha.keys())
    n_communities = len(all_communities)
    n_alphas = len(alpha_labels)

    # Bar positions
    bar_width = 0.25
    group_gap = 0.1
    x = np.arange(n_communities) * (n_alphas * bar_width + group_gap)

    bar_colors = ['r', 'b', 'g']

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (alpha_label, color) in enumerate(zip(alpha_labels, bar_colors)):
        vals = [community_results[alpha_label].get(c, 0) for c in all_communities]
        offsets = x + i * bar_width
        bars = ax.bar(offsets, vals, bar_width, color=color,
                      edgecolor='black', label=alpha_label)
        # Annotate bar values
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=12)

    # Add SVS annotations on x-axis labels
    svs_community = {}
    for node_idx, svs_val in svs_dict.items():
        if node_idx in rep_ind:
            resource = rep_ind[node_idx]
            comm = pipeline_to_community.get(resource)
            if comm:
                svs_community[comm] = svs_val

    tick_labels = [f"{c}\nSVS={svs_community.get(c, '')}" for c in all_communities]
    ax.set_xticks(x + bar_width * (n_alphas - 1) / 2)
    ax.set_xticklabels(tick_labels, fontsize=16)
    # Dynamic y-axis: max bar value + 25% headroom so labels always fit
    all_vals = [v for pc in community_results.values() for v in pc.values()]
    max_val = max(all_vals) if all_vals else 3.0
    ax.set_ylim(0, max_val * 1.25)

    ax.set_ylabel('Total Water Available / Total Demand', fontsize=fontsize)
    ax.set_xlabel('', fontsize=fontsize)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save with primary_alpha label in filename so each run produces a
    # separately named file — makes it easy to compare runs side by side.
    suffix = f'_{primary_alpha}' if primary_alpha else ''
    plt.savefig(f'water_availability_grouped{suffix}.png', dpi=300, bbox_inches='tight')
