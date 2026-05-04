# Computes Kendall's Tau correlation between the three repair orderings
# to produce Table II in the paper.
# No Monte Carlo needed - repair orderings are deterministic.
# Usage: python compute_kendall_tau.py

import sys
import os
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import (
    get_action_rule_metrics,
    get_action_rule_metrics_weighted,
    update_graph,
    find_paths_with_same_type_and_community,
    find_paths_to_failed_nodes,
    get_all_paths_index,
    get_power_action,
)
from calculate_rank_similarity import compute_correlation_matrix

GRAPH_PATH   = "data/community_hfg_model.graphml"
FAILED_NODES = ["Powerline2", "Powerline7", "Powerline12"]
RATIO        = [0.4, 0.6]
BFS_START    = "generates 'power' at 'Generator1'"

print("Loading graph...")
G = nx.read_graphml(GRAPH_PATH)
print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# build failed_indices the same way execute_simulation does
failed_indices = []
for idx, (node, data) in enumerate(G.nodes(data=True)):
    if node.split()[-1][1:-1] in FAILED_NODES:
        failed_indices.append(idx)

print(f"Failed node indices: {failed_indices}")

# build consumer_path_dict
paths_dict = find_paths_with_same_type_and_community(G, failed_indices)

consumer_path_dict = {}
for each in paths_dict.keys():
    consumer_path_dict[each] = set()
    for cons in paths_dict[each]:
        all_indices = [G.nodes[node]['index'] for node in cons]
        consumer_path_dict[each].update(all_indices)

# build correct_paths
all_paths_index       = get_all_paths_index(paths_dict, G)
paths_to_failed       = find_paths_to_failed_nodes(G, [0], failed_indices)
paths_to_failed_index = list(set(get_all_paths_index(paths_to_failed, G)))
correct_paths         = sorted(set(paths_to_failed_index + all_paths_index))

print(f"Relevant subgraph: {len(correct_paths)} nodes")

# compute repair orderings for each alpha
print("Computing repair orderings...")

G_c = update_graph('c', G, consumer_path_dict)
action_c = get_action_rule_metrics(G_c, BFS_START, "Criticality", 'c', consumer_path_dict)
order_criticality = get_power_action(action_c, correct_paths)

G_s = update_graph('s', G, consumer_path_dict)
action_s = get_action_rule_metrics(G_s, BFS_START, "SVS", 's', consumer_path_dict)
order_svs = get_power_action(action_s, correct_paths)

G_sc = update_graph('sc', G, consumer_path_dict)
action_sc = get_action_rule_metrics_weighted(G_sc, BFS_START, "SVS", "Criticality", RATIO, 'sc', consumer_path_dict)
order_weighted = get_power_action(action_sc, correct_paths)

# compute kendall tau
results = compute_correlation_matrix(order_svs, order_criticality, order_weighted)

tau_svs_crit    = results[('SVS', 'Criticality')][0]
tau_svs_weight  = results[('SVS', 'Weighted')][0]
tau_crit_weight = results[('Criticality', 'Weighted')][0]

table = f"""
TABLE II - Correlation of the Repair Order Rankings

                             Total Order    Total Order    Total Order
                             by SVS         by Criticality by SVS & Criticality
Total Order by SVS              1.00           {tau_svs_crit:.2f}           {tau_svs_weight:.2f}
Total Order by Criticality      {tau_svs_crit:.2f}           1.00           {tau_crit_weight:.2f}
Total Order by SVS
  and Criticality               {tau_svs_weight:.2f}           {tau_crit_weight:.2f}           1.00
"""
print(table)

# save results to file
output_file = "kendall_tau_results.txt"
with open(output_file, "w") as f:
    f.write("Kendall's Tau Correlation Between Repair Orderings\n\n")
    f.write(f"SVS vs Criticality:        tau = {tau_svs_crit:.4f}\n")
    f.write(f"SVS vs Weighted:           tau = {tau_svs_weight:.4f}\n")
    f.write(f"Criticality vs Weighted:   tau = {tau_crit_weight:.4f}\n")
    f.write(table)

print(f"Results saved to {output_file}")
