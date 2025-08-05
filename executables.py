## all functions in here should ideally take little more parameters than a restoration network:
# most changes should be made in the constants file

from RestorationNetwork import RestorationNetwork
from load_restoration_strategy_functions import superposition_regard_load, restoration_dispatch_evaluation, \
    resilience_indicator_strategy, strategy_scenario_comparison, compile_edc_indicators_for_load_restoration
from optimisation_strategy_functions import perform_optimisation, find_optimal_new_location
from network_evaluation import (get_ranked_distances, compare_modes_of_betweenness_centrality,
                                compare_modes_of_closeness_centrality, compare_modes_of_degree_centrality)
from visualisation import shortest_distance_matrix_heatmap, plot_restored_generation, visualise_centrality, \
    visualise_electric_degree_centrality, visualise_path_deviation
from helper_functions import *
from constants import *


def run_shortest_distances_matrix(restoration_network: RestorationNetwork, load_nodes_only=True, ranked_lengths=False,
                                  save=SAVE):
    """
    :param restoration_network:
    :param load_nodes_only: only visualise nodes with loads attached
    :param ranked_lengths: use the power station ranks (from closest to farthest) instead of the path lengths
    :param save:
    :return: go.Figure shortest distances heatmap
    """
    print_me('____________________ Start:  run_shortes_paths_heatmap ____________________')

    sdm_df = restoration_network.shortest_distance_matrix[restoration_network.load_nodes] if load_nodes_only \
        else restoration_network.shortest_distance_matrix
    sdm_df = get_ranked_distances(sdm_df) if ranked_lengths else sdm_df

    sdm_fig = shortest_distance_matrix_heatmap(sdm_df, ranked_lengths=ranked_lengths, save_as=save)

    if save:
        dataframe_to_file(sdm_df, keyword='shortest_distances_matrix_SDM')
    print_me('____________________ End:    run_shortes_paths_heatmap ____________________')
    return sdm_fig


def run_superposition_restoration(restoration_network: RestorationNetwork, visualise=True,
                                  identifier='base_scenario_superposition', save=SAVE):
    print_me('____________________ Start:  run_superposition_restoration ____________________')
    SCENARIO_PARAMETERS.update({'strategy': 'Superposition'})
    restoration_network.set_load_case(LOAD_CASE, scale=True)

    dispatched_power = superposition_regard_load(restoration_network)

    solution_dict = restoration_dispatch_evaluation(restoration_network, dispatched_power)
    print_me(solution_dict)

    if visualise:
        dispatch_fig = plot_restored_generation(dispatched_power, last_data=restoration_network.load_series,
                                                relative_values=restoration_network.load_factor)

    save_as_json(dispatched_power, restoration_network.shortest_distance_matrix, solution_dict, SCENARIO_PARAMETERS,
                 filepath=RESULT_FILEPATH, dataset_identifier=identifier)

    if save:
        dataframe_to_file(dispatched_power, keyword='VWA_dispatch_Superposition')

    print_me('____________________ End:  run_superposition_restoration ____________________')
    return dispatched_power


def run_resilience_indicator_restoration(restoration_network: RestorationNetwork, visualise=True,
                                         identifier='base_scenario_resilienzindikatoren', save=SAVE):
    print_me('____________________ Start:  run_resilience_indicator_restoration class version ____________________')
    SCENARIO_PARAMETERS.update({'strategy': 'Resilienzindikatoren'})

    restoration_network.set_load_case(LOAD_CASE, scale=True)
    edc_gen, edc_load = compile_edc_indicators_for_load_restoration(restoration_network)
    dispatched_power = resilience_indicator_strategy(restoration_network, generation_indicator=edc_gen,
                                                     load_indicator=edc_load)

    solution_dict = restoration_dispatch_evaluation(restoration_network, dispatched_power)
    print_me(solution_dict)

    if visualise:
        dispatch_fig = plot_restored_generation(dispatched_power, strategy='Resilienzindikatoren',
                                                last_data=restoration_network.load_series,
                                                relative_values=restoration_network.load_factor,
                                                save_as=save)

    save_as_json(dispatched_power, restoration_network.shortest_distance_matrix, solution_dict, SCENARIO_PARAMETERS,
                 filepath=RESULT_FILEPATH, dataset_identifier=identifier)

    if save:
        dataframe_to_file(dispatched_power, keyword='VWA_dispatch_Superposition')

    print_me('_______________________ End:  run_resilience_indicator_restoration ________________________')
    return dispatched_power


def run_optimisation_restoration(restoration_network: RestorationNetwork, visualise=True,
                                 dataset_identifier='base_scenario_optimierung', save=SAVE):
    print_me('____________________ Start:  run_optimisation_restoration ____________________')
    SCENARIO_PARAMETERS.update({'strategy': 'optimisation'})
    SCENARIO_PARAMETERS['kwargs'].update({'slack': SLACK})
    restoration_network.set_load_case(LOAD_CASE, scale=True)

    solution_dict = {}
    dispatched_power = perform_optimisation(restoration_network, solution_dict=solution_dict)
    solution_dict.update(restoration_dispatch_evaluation(restoration_network, dispatched_power))
    print_me(solution_dict)

    if visualise:
        fig = plot_restored_generation(dispatched_power, strategy='Optimierung',
                                       last_data=restoration_network.load_series,
                                       load_names=LOAD_CASE, relative_values=restoration_network.load_factor,
                                       save_as=save)

    save_as_json(dispatched_power, restoration_network.shortest_distance_matrix, SOLUTION_DICT, SCENARIO_PARAMETERS,
                 filepath=RESULT_FILEPATH, dataset_identifier=dataset_identifier)

    if save:
        dataframe_to_file(dispatched_power, keyword='VWA_dispatch_Resilienzindikatoren')

    print_me('____________________ End:  run_optimisation_restoration ____________________')
    return dispatched_power


def run_find_optimised_building_location(restoration_network: RestorationNetwork, visualise=True,
                                         dataset_identifier=None, save=SAVE):
    print_me('____________________ Start:  run_optimisation_restoration ____________________')
    SCENARIO_PARAMETERS.update({'strategy': 'optimisation'})
    SCENARIO_PARAMETERS['kwargs'].update({'slack': SLACK})
    restoration_network.set_load_case(LOAD_CASE, scale=True)

    solution_dict = {}
    location, dispatched_power = find_optimal_new_location(restoration_network)
    solution_dict.update(restoration_dispatch_evaluation(restoration_network, dispatched_power))
    print_me(location)
    print_me(solution_dict)

    if visualise:
        fig = plot_restored_generation(dispatched_power, strategy='Optimierung',
                                       last_data=restoration_network.load_series,
                                       load_names=LOAD_CASE, relative_values=restoration_network.load_factor,
                                       save_as=save)

    save_as_json(dispatched_power, restoration_network.shortest_distance_matrix, SOLUTION_DICT, SCENARIO_PARAMETERS,
                 filepath=RESULT_FILEPATH, dataset_identifier=dataset_identifier)

    if save:
        dataframe_to_file(dispatched_power, keyword='VWA_dispatch_Optimierung')

    print_me('____________________ End:  run_optimisation_restoration ____________________')
    return dispatched_power


def run_betweenness_centrality(restoration_network: RestorationNetwork, visualise=True, save=SAVE):
    print('____________________ Start:  run_betweenness_centrality ____________________')
    bc_comp = compare_modes_of_betweenness_centrality(restoration_network.network_graph,
                                                      restoration_network.generators)

    if visualise:
        fig = visualise_centrality(bc_comp, save_as=save)

    if save:
        dataframe_to_file(bc_comp, keyword='resilienzindikator_BC_PTBC')
    print('____________________ End:  run_betweenness_centrality ____________________')
    return bc_comp


def run_closeness_centrality(restoration_network: RestorationNetwork, visualise=True, save=SAVE):
    print('____________________ Start:  run_closeness_centrality ____________________')
    cc_comp = compare_modes_of_closeness_centrality(restoration_network)

    if visualise:
        fig = visualise_centrality(cc_comp, mode='Closeness', save_as=save)

    if save:
        dataframe_to_file(cc_comp, keyword='resilienzindikator_BC_PTBC')
    print('____________________ End:  run_closeness_centrality ____________________')
    return cc_comp


def run_degree_centrality(restoration_network: RestorationNetwork, visualise=True,
                          save=SAVE):
    print('____________________ Start:  run_degree_centrality ____________________')
    dc_comp = compare_modes_of_degree_centrality(restoration_network)

    if visualise:
        fig = visualise_electric_degree_centrality(dc_comp, save_as=save)

    if save:
        dataframe_to_file(dc_comp, keyword='resilienzindikator_BC_PTBC')
    print('____________________ End:  run_degree_centrality ____________________')


def run_scenario(restoration_network, node, build=False, shutdown=False, append_scenario=False, exec_strategy='all',
                 visualise=True, save=False):
    """
    apply the specified scenario (either shutdown or new build at given node)
    and run the specified restoration strategies
    :param restoration_network:
    :param node: which node is affected
    :param build: bool is a new power station being built
    :param shutdown: bool is an existing power station being shutdown
    :param append_scenario: is this new scenario supposed to override any scenarios that might currently be in place or
                            do you want to append it and analyse the combination of scenarios
    :param exec_strategy: ['all', 'optimierung', 'resilienzindikatoren', 'superposition'] specify which restoration
                            strategy is to be applied to the scenario
    :param visualise: visualise the difference in path lengths between applied and base scenario;
    BASE SCENARIO MUST HAVE BEEN RUN AND SAVED TO THE JSON FILE WITH THE STANDARD IDENTIFIER FOR THIS TO WORK
    :param save:
    :return:
    """
    if shutdown == build:
        print('Scenario application. Specify either new build or shutdown.')
        return 1
    shutdown_scenario = {SCENARIO_NAME: f'Abschaltung_{node}',
                         SCENARIO_LOCATION: node,
                         SCENARIO_GENERATION_VALUE: 0,
                         SCENARIO_DESCRIPTION: f'Kohleausstieg {node}'}
    new_build_scenario = {SCENARIO_NAME: f'Neubau_{node}',
                          SCENARIO_LOCATION: node,
                          SCENARIO_GENERATION_VALUE: STANDARD_POWER,
                          SCENARIO_DESCRIPTION: f'Neubau Kraftwerk {node}'}
    scenario_dict = new_build_scenario if build else shutdown_scenario
    restoration_network.set_load_case(LOAD_CASE, scale=True)
    restoration_network.apply_scenario(scenario_dict, append_scenario=append_scenario,
                                          overwrite_prev=not append_scenario)

    if (exec_strategy == 'superposition') | (exec_strategy == 'all'):
        run_superposition_restoration(restoration_network, visualise=False,
                                      identifier=f'{scenario_dict[SCENARIO_NAME]}_superposition')
        if visualise:
            visualise_path_deviation(retrieve_relevant_comparison_data(f'base_scenario_superposition',
                                                                       RESULT_FILEPATH),
                                     optimum=restoration_network.get_closest_power_stations(True), strategy='superposition')
    if (exec_strategy == 'resilienzindikatoren') | (exec_strategy == 'all'):
        run_resilience_indicator_restoration(restoration_network, visualise=False,
                                             identifier=f'{scenario_dict[SCENARIO_NAME]}_resilienzindikatoren')
        if visualise:
            visualise_path_deviation(retrieve_relevant_comparison_data(f'base_scenario_resilienzindikatoren',
                                                                       RESULT_FILEPATH),
                                     optimum=restoration_network.get_closest_power_stations(True), strategy='resilienzindikatoren')
    if (exec_strategy == 'optimierung') | (exec_strategy == 'all'):
        run_optimisation_restoration(restoration_network, visualise=False,
                         dataset_identifier=f'{scenario_dict[SCENARIO_NAME]}_optimierung')
        if visualise:
            visualise_path_deviation(retrieve_relevant_comparison_data(f'base_scenario_optimierung',
                                                                       RESULT_FILEPATH),
                                     optimum=restoration_network.get_closest_power_stations(True), strategy='optimierung')


    res = strategy_scenario_comparison(restoration_network,
                                       scenario_type='Neubau' if build else 'Abschaltung',
                                       scenario_nodes=[node],
                                       make_relative=False, use_mean=False, save=save)

    return res
