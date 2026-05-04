from scipy.stats import rankdata
from scipy.stats import kendalltau


def get_similarity_ranks(list0, list1):
    # compute the ranks for each list
    ranks_list1 = rankdata(list0)
    ranks_list2 = rankdata(list1)
    tau, p_value = kendalltau(ranks_list1, ranks_list2)
    return tau, p_value


def compute_correlation_matrix(full_order_svs, full_order_criticality, full_order_weighted):
    # compute all pairwise Kendall's Tau correlations between the three repair orderings
    # produces the 3x3 correlation table for Table II in the paper
    pairs = {
        ('SVS', 'Criticality'):      (full_order_svs, full_order_criticality),
        ('SVS', 'Weighted'):         (full_order_svs, full_order_weighted),
        ('Criticality', 'Weighted'): (full_order_criticality, full_order_weighted),
    }

    print("\nKendall's Tau Correlation Matrix:")
    print(f"{'Pair':<35} {'tau':>8} {'p-value':>12}")
    print("-" * 57)

    results = {}
    for (a, b), (l1, l2) in pairs.items():
        tau, p_value = get_similarity_ranks(l1, l2)
        results[(a, b)] = (tau, p_value)
        print(f"{a} vs {b:<25} {tau:>8.4f} {p_value:>12.4f}")

    return results
