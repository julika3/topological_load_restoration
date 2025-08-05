import pandas as pd
import networkx as nx
from math import e
from RestorationNetwork import RestorationNetwork


def get_ranked_distances(distance_to_ps):
    ps_nodes_rank = distance_to_ps.copy()
    for ss in distance_to_ps.columns:
        ps_nodes_rank[ss] = distance_to_ps[ss].rank(method='min')
    return ps_nodes_rank


def minimum_possible_distance(distance_to_ps, normalised=True):
    min_distances = distance_to_ps.min(0)
    min_distance_total = distance_to_ps.min(0).sum()
    if normalised:
        min_distance_total = min_distance_total / len(min_distances.index)
    return min_distance_total


def single_source_all_shortest_paths(graph, source, target_iterator):
    """
    :param graph: networx graph
    :param source: node from which shortes paths are examined
    :param target_iterator: list of nodes to which shortest paths are found
    :return:
    """
    shortest_paths_dict = {}
    pred = nx.predecessor(graph, source)
    for node in target_iterator:
        shortest_paths_dict[node] = list(nx.algorithms.shortest_paths.generic._build_paths_from_predecessors(
            [source], node, pred))
    return shortest_paths_dict


def shortest_paths_to_subset(graph, targets, exclude_target):
    """

    :param graph: networkx graph type network model
    :param targets: subset of nodes
    :return: shortest_paths_dict: keys: all nodes from G that aren't in targets subset
                                  items: dictionary of the shortest paths to all nodes in targets
    """
    shortest_paths_dict = {}

    for n in graph.nodes:
        if exclude_target and (n != 'Thyrow') and (n in targets):
            # don't calculate bc for nodes to which the shortest paths are regarded
            continue
        sp_dict = single_source_all_shortest_paths(graph, n, targets)
        shortest_paths_dict.update({n: sp_dict})

    return shortest_paths_dict


def power_transfer_betweenness_centrality(graph, subset, exclude_subset_to_subset=False):
    """
    :param exclude_subset_to_subset: 
    :param graph: networkx graph type network model
    :param subset: list of nodes to which all shortest paths from other nodes are regarded
    :return: dict with betweenness centrality score for all graph nodes not in subset
    """

    betweenness = {}
    shortest_paths_dict = shortest_paths_to_subset(graph, subset, exclude_subset_to_subset)

    # analog to the normalisation of regular betweenness centrality (see networkx docs)
    normalisation_factor = 1 / ((graph.order() - 1) * (len(subset) - 2))

    for n1 in shortest_paths_dict.keys():
        betweenness_centrality = 0
        for n2, sp_dict in shortest_paths_dict.items():
            if n2 == n1:
                continue  # n1 is on all shortest paths connecting n1; this is therefore disregarded
            for target, paths in sp_dict.items():
                score = 0  # counts on how many shortest paths n1 appears
                n_paths = len(paths)  # counts the total number of possible shortest paths
                for p in paths:
                    if (n1 in p) & (p[-1] != n1):
                        # shortest path that passes through n1
                        score += 1
                betweenness_centrality += score / n_paths if score > 0 else 0

        betweenness.update({n1: betweenness_centrality * normalisation_factor})

    return betweenness


def compare_modes_of_betweenness_centrality(graph, subset):
    bc_compare = pd.DataFrame.from_dict(nx.betweenness_centrality(graph), orient='index',
                                        columns=['Betweenness Centrality'])
    bc_compare['Power Transfer<br>Betweenness Centrality'] = pd.DataFrame.from_dict(
        power_transfer_betweenness_centrality(graph, subset, False), orient='index')[0]
    return bc_compare


def power_transfer_closeness_centrality(restoration_network: RestorationNetwork):
    """

    :param restoration_network:
    :return:
    """
    def ptcc(n, subset_to, dual):
        n_subset = len(subset_to) - 1 if dual else len(subset_to)
        val = 0
        for s in subset_to:
            val += nx.shortest_path_length(restoration_network.network_graph, source=n, target=s)
        val = n_subset / val
        return val

    ptcc_dict = {}
    for n in restoration_network.network_graph.nodes():
        dual = n in restoration_network.dual_substations
        if n in restoration_network.generators:
            subset_to = restoration_network.load_nodes
            name = f'{n}' if dual else n
            ptcc_dict.update({name: ptcc(n, subset_to, dual)})
        if n in restoration_network.load_nodes:
            subset_to = restoration_network.generators
            name = f'{n}_L' if dual else n
            ptcc_dict.update({name: ptcc(n, subset_to, dual)})

    return ptcc_dict


def compare_modes_of_closeness_centrality(restoration_network: RestorationNetwork):
    cc_compare = pd.DataFrame.from_dict(power_transfer_closeness_centrality(restoration_network), orient='index',
                                        columns=['Power Transfer<br>Closeness Centrality'])
    cc_compare['Closeness Centrality'] = pd.DataFrame.from_dict(
        nx.closeness_centrality(restoration_network.network_graph), orient='index')[0]
    cc_compare = cc_compare.sort_index()
    return cc_compare


def electrical_degree_centrality(restoration_network: RestorationNetwork):
    edc_dict = {}
    S_Gmax = restoration_network.generation_series.max()

    for node in restoration_network.network_graph.nodes():
        adjacent = nx.ego_graph(restoration_network.network_graph, node, center=False)
        S_Li = restoration_network.load_series.to_dict().get(node, 0)
        S_Gi = restoration_network.generation_series.to_dict().get(node, 0)
        edc = (adjacent.order() / (restoration_network.network_graph.order() - 1)
               * e ** (- (S_Gmax - (S_Gi + S_Li)) / S_Gmax))
        edc_dict.update({node: edc})

    return edc_dict


def compare_modes_of_degree_centrality(restoration_network: RestorationNetwork):
    edc_dict = electrical_degree_centrality(restoration_network)
    dc_dict = nx.degree_centrality(restoration_network.network_graph)
    dc_compare = pd.DataFrame.from_dict(edc_dict, orient='index', columns=['EDC'])
    dc_compare['DC'] = pd.DataFrame.from_dict(dc_dict, orient='index')[0]
    return dc_compare
