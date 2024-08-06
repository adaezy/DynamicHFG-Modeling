
from scipy.stats import rankdata
from scipy.stats import kendalltau


def get_similarity_ranks(list0, list1):
    # Compute the ranks for each list
    ranks_list1 = rankdata(list0)
    ranks_list2 = rankdata(list1)

    tau, p_value = kendalltau(ranks_list1, ranks_list2)
    return tau, p_value


tau, p_value = get_similarity_ranks(full_order_weighted ,full_order_criticality )
print(tau,p_value)