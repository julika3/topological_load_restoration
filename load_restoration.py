import pandas as pd
import networkx as nx

from constants import *
from RestorationNetwork import RestorationNetwork
from helper_functions import *
from ValidationError import *
from network_evaluation import electrical_degree_centrality
from helper_functions import retrieve_relevant_comparison_data


def superposition_regard_load(restoration_network: RestorationNetwork,
                              threshold=LOAD_ESTIMATE):
    """
    in this restoration strategy power is distributed fairly between all nodes that can be reached with a path
    of distance 0; if power remains after all reached nodes have met their demands, the distance is increased by
    one step; if a node can simultaneously be supplied by two power stations they both supply an even share.
    for details refer to thesis
    :param restoration_network:
    :param threshold: maximum load to be given to a node if no demand is specified
    :return:
    """
    def get_gen_distribution(gen_share, remaining_load, is_recipient):
        gen_dist = is_recipient.mask(is_recipient != 0, 0)
        relevant_pairs = is_recipient[is_recipient].stack().index.tolist()
        for row, col in relevant_pairs:
            min_val = min(gen_share.loc[row], remaining_load.loc[col])
            gen_dist.loc[row, col] = min_val
        return gen_dist

    # start with an empty dataframe
    distributed_power = restoration_network.shortest_distance_matrix.mask(
        restoration_network.shortest_distance_matrix != 0, 0)
    remaining_gen = restoration_network.generation_series.copy()
    # check if load demand is met either by specified demand or threshold
    max_power = restoration_network.load_series.loc[distributed_power.columns] if restoration_network.load_series \
                                                                                  is not None else threshold

    for distance in range(0, restoration_network.shortest_distance_matrix.max(axis=None) + 1):

        continue_distributing = True
        while continue_distributing:
            remaining_load = max_power - distributed_power.sum(axis=0)
            # check if nodes have uncovered demand
            # creates a dataframe that contains bool values within a substation x power station dataframe
            is_not_covered = pd.DataFrame([remaining_load > 0], index=distributed_power.index)

            does_ps_have_power_left = pd.DataFrame({col: remaining_gen > 1 for col in
                                                    restoration_network.shortest_distance_matrix.columns},
                                                   index=restoration_network.shortest_distance_matrix.index)
            # subset of power stations that could supply a node at given distance and have undistributed power
            pot_suppliers = ((restoration_network.shortest_distance_matrix == distance) &
                             does_ps_have_power_left).sum(axis=0)
            suppliable_substations = (remaining_load > 0) & (pot_suppliers > 0)
            if suppliable_substations.sum() == 0:
                continue_distributing = False
                continue
            remaining_load = remaining_load / pot_suppliers.replace(0, 1)  # don't divide by zero

            is_recipient = ((restoration_network.shortest_distance_matrix == distance) & is_not_covered)
            recipients_count = is_recipient.sum(axis=1)

            gen_share = (remaining_gen / recipients_count.map(lambda x: 1 if not x else x)).replace(float('inf'), 0)
            gen_distr = get_gen_distribution(gen_share, remaining_load, is_recipient)

            # remove distributed power from remaining generation popwer
            remaining_gen -= gen_distr.sum(axis=1)

            # add the power distributed in this round to the overall power so far
            distributed_power = distributed_power + gen_distr

    validate_result(distributed_power, restoration_network.generation_series)
    distributed_power = pd.DataFrame(distributed_power).fillna(0).loc[restoration_network.generation_series.index]

    return distributed_power


def resilience_indicator_based_approach(restoration_network: RestorationNetwork):

    # dicts to store the dispatch results
    received_dict = {x: {'sum': 0} for x in restoration_network.load_nodes}
    distribution_dict = {x: {'remaining': restoration_network.generation_series.loc[x]} for x in restoration_network.generators}

    # Calculate electric degree centrality to define the order of
    # 1. Power stations distributing their power and
    # 2. Load nodes being chosen for dispatch
    edc_dict = electrical_degree_centrality(restoration_network)
    edc = pd.DataFrame.from_dict(edc_dict, orient='index')
    edc_gen = edc.loc[restoration_network.generators].sort_values([0], ascending=False)
    edc_load = edc.loc[restoration_network.load_nodes].sort_values([0], ascending=False)

    distance = 0
    max_distance_to_ps = restoration_network.shortest_distance_matrix.max().max()
    # Starting at the power stations (distance=0)
    # Move through network whilst increasing the path distance incrementally (distance += 1)
    while distance <= max_distance_to_ps:
        for ps in edc_gen.index:
            ps_power = distribution_dict[ps]['remaining']
            if ps_power == 0:  # all power has already been distributed
                continue

            # check which load nodes can be reached from the power station given the current distance
            path_lengths_ps = restoration_network.shortest_distance_matrix.transpose()[ps]
            connections = path_lengths_ps[path_lengths_ps == distance].index
            # rank the available connections via their edc value
            connections_iterator = edc_load.loc[[x for x in connections
                                                 if x in restoration_network.load_nodes]].index

            # distribute the power starting at the load_node with the highest edc value
            for con in connections_iterator:
                # determin the (remaining) lode demand of the current node
                ss_received = received_dict[con]['sum']
                demand = restoration_network.load_series.loc[con] - ss_received

                if demand > 0:
                    # cover as much of the demand as is possible give the remaining power
                    power_share = demand if demand < ps_power else ps_power
                    received_dict[con]['sum'] = received_dict[con]['sum'] + power_share
                    received_dict[con].update({ps: power_share})
                    ps_power -= power_share
                    distribution_dict[ps]['remaining'] = ps_power
                    distribution_dict[ps].update({con: power_share})
                    if ps_power == 0:
                        break

        distance += 1

    # turn result into dataframe
    dispatched_power_df = pd.DataFrame(received_dict).fillna(0)
    available_keys = sorted(dispatched_power_df.index.intersection(restoration_network.generators))
    dispatched_power_df = dispatched_power_df.loc[available_keys]

    # ensure that no problem occurred and the dispatched power equals the available power
    validate_result(dispatched_power_df, restoration_network.generation_series)

    return dispatched_power_df


def evaluate_restoration_result(restoration_network: RestorationNetwork, generation_distribution, mean=True,
                                mean_ss=False, scenario_paths=None):
    """
    determine the deviation from the result to the base scenario result
    :param restoration_network:
    :param generation_distribution:
    :param scenario_paths: need to be given if the result was loaded from file and scenario changes aren't stored in
                            restoration network
    :param mean: take mean path value
    :param mean_ss: calculate a mean of all the different power station paths that are used by a single node
    :return:
    """
    def normalised_distance_taken(uw, kw_array):
        """
        get distance to a supplying power station as well as the deviation in distance to the theoretically closest ps
        if mean_ss: get a mean value over all supplying power station distances else they are added up
        :param uw: substation name
        :param kw_array: supplying power stations
        :return:
        """
        path_lengths = restoration_network.shortest_distance_matrix if scenario_paths is None else scenario_paths
        dmin = restoration_network.shortest_distance_matrix_base[uw].min()
        d = 0
        n = len(kw_array)
        diff = 0
        for kw in kw_array:
            d_temp = path_lengths.loc[kw, uw]
            d += d_temp
            diff += d_temp - dmin if mean_ss else d_temp
        d = d / n if mean_ss else d
        diff = diff / n if mean_ss else diff - dmin
        return d, diff

    td = 0
    n2 = 0
    tdiff = 0
    for uw in generation_distribution.columns:
        supplier = generation_distribution[generation_distribution[uw] > 1][uw].index
        if len(supplier) == 0:
            continue
        nd, ndiff = normalised_distance_taken(uw, supplier)
        td += nd
        tdiff += ndiff
        n2 += 1

    if mean & (n2 > 0):
        td = td / n2
        tdiff = tdiff / n2

    return float(td), float(tdiff)


def compare_restoration_results(restoration_network: RestorationNetwork, base_case=None, comparison_cases=None,
                                strategy='', relative=False, mean=False):
    """
    creates a dataframe wherein results of different load_restoration runs are compared tp one another
    :param restoration_network:
    :param base_case: tuple containing (resulting substation/supplying_generator pairs after load restoration;
                                        path lengths between substations and generations in given scenario;
                                        scenario name)
    :param comparison_cases: list of tuples in the same format as base_case
    :param strategy: load restoration strategy that was used in all given cases
    :param relative: (bool) if True results of comparison cases are made relative to the base case in the dataframe
    :return: res_df dataframe
    """
    result1, paths1, scenario1 = base_case

    distances1 = paths1[result1 > 1]
    total1, tdiff1 = evaluate_restoration_result(restoration_network, result1, scenario_paths=paths1, mean=mean)

    res_df = pd.DataFrame(zip(['Alle', 'Veränderte', 'Alle', 'Veränderte'],
                              [strategy for i in range(4)],
                              [total1, float('nan'), tdiff1, float('nan')]),
                          columns=['Welche UW', 'VWA-Strategie', scenario1],
                          index=['Pfadlänge', 'Pfadlänge',
                                 'Abweichung zur optimalen Pfadlänge', 'Abweichung zur optimalen Pfadlänge'])

    if comparison_cases is None:
        res_df = res_df[res_df['Welche UW'] == 'Alle']
    else:
        for comparison_case in comparison_cases:
            result2, paths2, scenario2 = comparison_case
            distances2 = paths2[result2 > 1]
            scenario_paths = paths2 if 'Neubau' in scenario2 else None
            total2, tdiff2 = evaluate_restoration_result(restoration_network, result2, scenario_paths=scenario_paths,
                                                         mean=mean)

            changed_path_lengths = distances2.sub(distances1, axis='index', fill_value=0)
            changed = changed_path_lengths.sum()[changed_path_lengths.sum() != 0].index.tolist()

            total2_c, tdiff2_c = evaluate_restoration_result(restoration_network, result2[changed], scenario_paths=paths2,
                                                             mean=mean)
            scenario_vals = [total2, total2_c, tdiff2, tdiff2_c]

            if relative:
                total1_c, tdiff1_c = evaluate_restoration_result(restoration_network, result1[changed],
                                                                 scenario_paths=paths1, mean=mean)
                scenario_vals = [total2/total1, total2_c/total1_c, tdiff2/tdiff1, tdiff2_c/tdiff1_c]

            res_df[f'{scenario2.split("_")[1]}_{scenario2.split("_")[0]}'] = scenario_vals

        if not relative:
            res_df = res_df[res_df['Welche UW'] == 'Alle']

    return res_df.round(4)


def validate_result(received_df, generation_series):
    """
    check whether a load_restoration result can be achieved given the actual available generation
    :param received_df: power shares as they are distributed between the substations
    :param generation_series: generators and their available power
    :return: bool
    """
    generation_distributed = received_df.sum(axis=1)
    gen_diff = generation_series.sub(generation_distributed, fill_value=0)
    check = len(gen_diff[gen_diff < -1].index) == 0

    if check:
        print_me('Distributed power was validated and does not exceed available generation.')
    else:
        raise ValidationError('Validation unsuccessful. Distributed power exceeds available generation.')
    return check


def restoration_dispatch_evaluation(restoration_network, dispatched_power):
    """
    Evaluates the effectiveness of a restoration dispatch strategy.

    Parameters:
    :param restoration_network: Network object containing restoration parameters and load factor.
    :param dispatched_power: Dictionary or structure representing the dispatched distributed generation.
    :return: solution_dict (dict): Dictionary containing metrics such as load factor, total and mean path lengths,
      and deviations from ideal restoration paths.
    """
    # Evaluate total path length and deviation without averaging
    total_path_length, total_path_deviation = evaluate_restoration_result(
        restoration_network, dispatched_power, mean=False, mean_ss=False
    )

    # Evaluate mean path length and deviation across all substations
    mean_path, mean_path_deviation = evaluate_restoration_result(
        restoration_network, dispatched_power, mean_ss=False
    )

    # Compile results into a dictionary
    solution_dict = {
        'load_factor': restoration_network.load_factor,
        'total_path_length': total_path_length,
        'total_path_deviation': total_path_deviation,
        'mean_path_length': mean_path,
        'mean_path_deviation': mean_path_deviation
    }

    return solution_dict


def strategy_scenario_comparison(restoration_network, scenario_type='base_case', scenario_nodes=None,
                                 make_relative=False, use_mean=False, save=True):
    """
    compares the load restoration results of all three strategies for different scenarios.
    the scenario calculation must have been executed for every scenario_type + node combination with the result stored
    in RESULT_FILEPATH; if run_scenario was executed results are automatically stored in the right place
    :param restoration_network:
    :param scenario_type: 'Neubau' (new build) or 'Abschaltung' (shut down)
    :param scenario_nodes: list of nodes to be included in the comparison
    :param make_relative: make values of comparison cases relative to the base_case results
    :param use_mean: returns mean path instead of total path value
    :param save:
    :return:
    """
    comp_test = pd.DataFrame()
    for vwa in ['optimierung', 'resilienzindikatoren', 'superposition']:
        base_case = retrieve_relevant_comparison_data(f'base_scenario_{vwa}', RESULT_FILEPATH)
        comparison_cases = None if scenario_type == 'base_case' else [
            retrieve_relevant_comparison_data(f'{scenario_type}_{node}_{vwa}', RESULT_FILEPATH)
            for node in scenario_nodes]
        comp_test_temp = compare_restoration_results(restoration_network,
                                                     base_case=base_case,
                                                     comparison_cases=comparison_cases,
                                                     strategy=vwa, relative=make_relative, mean=use_mean)
        comp_test = pd.concat([comp_test, comp_test_temp])

    if save:
        comp_test.to_excel(f'results/VWA-Strategien_Vergleich_{scenario_type}_{LOAD_CASE}'
                           f'{"_mean" if use_mean else "_total"}'
                           f'{"_relative" if make_relative else ""}.xlsx')

    return comp_test
