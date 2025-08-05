import pandas as pd
import networkx as nx

from constants import *
from helper_functions import print_me, generate_color_dict


class RestorationNetwork:
    def __init__(self, network_model_name):
        self.network_model_name = network_model_name
        # Netzmodell Datenbasis
        lines_dataframe = pd.read_excel(f'data/{NETWORK_MODELS[network_model_name]["filepath_network"]}')
        # list of substations with further information including their NWAR and power if a power station is connected
        self.substation_dataframe = pd.read_excel(f'data/'
                                                  f'{NETWORK_MODELS[network_model_name]["filepath_substation_infos"]}'
                                                  ).set_index(SUBSTATION_COLUMN)
        self.network_graph = nx.from_pandas_edgelist(lines_dataframe,
                                                     source=SUBSTATION_NAME_1, target=SUBSTATION_NAME_2,
                                                     create_using=nx.Graph)

        self.graph_order = self.network_graph.order()

        # dataframe mit allen Erzeugern und ihrer Erzeugungsleistung
        self.generation_series = self.substation_dataframe[self.substation_dataframe[POWER_COLUMN] > 0][
            POWER_COLUMN].sort_index()

        self.generators = self.generation_series.index.tolist()
        if NODE_TYPE_FILTER is not None:
            self.substations = list(set(self.substation_dataframe[self.substation_dataframe[TYPE_COLUMN] ==
                                                                  NODE_TYPE_FILTER].index) &
                                set(self.network_graph.nodes()))
        else:
            self.substations = self.network_graph.nodes()
        self.dual_substations = None

        self.load_case = None
        self.load_series = None
        self.max_load = None
        self.load_factor = None
        self.load_nodes = None

        self.shortest_distance_matrix_base = self.get_shortest_distance_matrix()
        self.shortest_distance_matrix = self.shortest_distance_matrix_base.copy()

        self.generator_colors = generate_color_dict(self.generators)

        self.scenario_applied = 'Basisszenario'
        self.scaled = False
        self.manipulated = False
        self.sdm_manipulated = None
        self.load_series_manipulated = None

    def scenario_simulation(self, scenario_name):
        if NETWORK_MODELS[self.network_model_name]['filepath_scenario'] is None:
            print('No scenarios given. Please append filepath for scenarios in constants.py')
        else:
            scenario_file = pd.read_excel(f'data/{NETWORK_MODELS[self.network_model_name]["filepath_scenario"]}',
                                          index_col=SCENARIO_NAME)
            scenarios = scenario_file.index.to_list()
            if scenario_name not in scenarios:
                print('Scenario unavailable')

    def reset_previous_changes(self):
        """reset any scenarios that have been applied"""
        self.generation_series = self.substation_dataframe[self.substation_dataframe[POWER_COLUMN] > 0][
            POWER_COLUMN].sort_index()
        self.generators = self.generation_series.index.tolist()
        self.shortest_distance_matrix = self.get_shortest_distance_matrix()
        self.sdm_manipulated = None
        self.load_series_manipulated = None
        self.manipulated = False
        self.scenario_applied = 'Basisszenario'

    def apply_scenario(self, scenario_dict, overwrite_prev=False, append_scenario=False):
        """
        makes changes to the model to simulate the specified scenario;
        changes are made in a way that can be reset without reloading the network
        :param scenario_dict: key value pairs for the following: SCENARIO_NAME, SCENARIO_LOCATION,
                                SCENARIO_GENERATION_VALUE, SCENARIO_DESCRIPTION. keys are stored in constants.py
        :param overwrite_prev: if a scenario is applied already should it be reset
        :param append_scenario: should the specified scenario be modelled additionally to whatever is currently applied
        :return:
        """
        if self.scenario_applied != 'Basisszenario':
            if overwrite_prev:
                self.reset_previous_changes()
            elif append_scenario:
                print('Current scenario', self.scenario_applied, 'will be used as basis.')
            else:
                print('Error. Existing manipulations would be overwritten')
                return 1

        self.scenario_applied = f'{self.scenario_applied}; {scenario_dict[SCENARIO_NAME]}' if \
            self.scenario_applied != 'Basisszenario' else scenario_dict[SCENARIO_NAME]
        SCENARIO_PARAMETERS['kwargs']['scenario_descriptor'] = self.scenario_applied
        if scenario_dict[SCENARIO_LOCATION] in self.generators:
            # generation value is changed, increase generation of power station accordingly
            self.generation_series.loc[scenario_dict[SCENARIO_LOCATION]] = (self.generation_series.loc[
                                                                                scenario_dict[SCENARIO_LOCATION]] +
                                                                            scenario_dict[SCENARIO_GENERATION_VALUE])
        else:  # generator is added
            self.generation_series = pd.concat([self.generation_series,
                                                pd.Series({scenario_dict[SCENARIO_LOCATION]:
                                                               scenario_dict[SCENARIO_GENERATION_VALUE]})])
        self.generation_series = self.generation_series[self.generation_series > 0].sort_index()
        self.generators = self.generation_series.index.tolist()
        self.shortest_distance_matrix = self.get_shortest_distance_matrix()

        if (self.load_case is not None) & self.scaled:  # update scaling of loads and load_factor
            self.set_load_case(self.load_case, self.scaled)
        else:
            self.load_factor = self.generation_series.sum() / self.load_series.sum()

        print(f'{scenario_dict[SCENARIO_NAME]} applied')

    def get_shortest_distance_matrix(self):
        """
        calculate the shortest path for every power station - load node combination for the class own network_graph
        and create a dataframe
        """
        distance_to_ps = pd.DataFrame(self.generators, columns=['Power_Stations'])

        graph = self.network_graph

        for ss in graph.nodes:
            if (ss in self.generators) & (ss not in self.substations):
                continue
            distance_tuples = []
            ps_nodes = []
            for ps in self.generators:
                # for substation ss check the shortest path to power station ps
                try:
                    nodes = nx.dijkstra_path_length(graph, ss, ps)
                except nx.NetworkXNoPath:
                    nodes = 999
                distance_tuples.append((ps, nodes))
                ps_nodes.append(nodes)

            res_temp = pd.DataFrame(zip(self.generators, ps_nodes), columns=['Power_Stations', ss])
            # add the distances (in nodes) to the result df
            distance_to_ps = pd.concat([distance_to_ps, res_temp[[ss]]], axis=1)

        distance_to_ps = distance_to_ps.set_index('Power_Stations')
        return distance_to_ps

    def get_ranked_distances(self):
        ps_nodes_rank = self.shortest_distance_matrix.copy()
        for ss in self.shortest_distance_matrix.columns:
            ps_nodes_rank[ss] = self.shortest_distance_matrix[ss].rank(method='min')
        return ps_nodes_rank

    def minimum_possible_distance(self, normalised=True):
        """minimum path length if every load node was supplied by its closest power station"""
        min_distances = self.shortest_distance_matrix.min(0)
        min_distance_total = self.shortest_distance_matrix.min(0).sum()
        if normalised:
            min_distance_total = min_distance_total / len(min_distances.index)
        return min_distance_total

    def scale_loads(self):
        """if the load_factor (versorgungsgrad) is less than one this scaling can be used to assign the load share"""
        if self.load_factor is None:
            print('Keine Lasten zum skalieren vorhanden.')
        elif self.load_factor > 1:
            print_me('Lastfaktor beträgt {:.4f}. Lasten werden nicht hockskaliert.'.format(self.load_factor))
        else:
            print_me('Lastfaktor beträgt {:.4f}. Lasten werden runterskaliert.'.format(self.load_factor))
            self.load_series = self.load_series * self.load_factor
            self.scaled = True

    def set_load_case(self, load_case, scale=False):
        """
        apply the load values as specified in the substation_dataframe
        :param load_case: key value for the statistical value; this should be the column header in the file
        :param scale: bool should loads be scaled if the total load exceeds total generation
        :return:
        """
        if load_case not in self.substation_dataframe:
            print('Lastfall nicht vorhanden')
            return 1
        self.load_case = load_case
        self.load_series = self.substation_dataframe[self.load_case].fillna(0)
        self.max_load = self.load_series.max()
        self.load_factor = self.generation_series.sum() / self.load_series.sum()
        self.dual_substations = self.load_series[self.load_series > 0].index.intersection(
            self.generation_series.index).tolist()
        self.load_nodes = list(set(self.load_series[self.load_series > 0].index) &
                               set(self.network_graph.nodes()))
        if scale:
            self.scale_loads()

    def reset_loads(self):
        if self.scaled:
            self.load_series = self.load_series / self.load_factor
            self.scaled = False
            print('Lasten zurückgesetzt')
        else:
            print('Lasten befinden sich bereits im Ausgangszustand')

    def handle_dual_substations(self):
        """give a nod with load and generation power a second virtual node that share all its properties so that
            a differentiated analysis of its function as load node or as a generator node is made possible"""
        self.sdm_manipulated = self.shortest_distance_matrix.copy()
        self.load_series_manipulated = (self.load_series[self.load_series.index.map(lambda x: x in self.substations)])

        if len(self.dual_substations) > 0:
            print('Helfsknoten für Knoten mit Last und Erzeugung werden erstellt.')
            for ss in self.dual_substations:
                self.sdm_manipulated.rename({ss: f'{ss}_load'}, axis=1, inplace=True)
                self.load_series_manipulated.rename({ss: f'{ss}_load'}, inplace=True)
            self.manipulated = True
        else:
            print('Keine Knoten mit Last und Erzeugung vorhanden.')


    def get_closest_power_stations(self, load_nodes_only=False):
        """
        determines the path distance to the closest power station for every node
        :param load_nodes_only: filter resulting dataframe to only include load nodes
        :return:
        """
        closest_ps = pd.DataFrame(self.shortest_distance_matrix_base.min()).transpose()
        closest_ps.index = ['Nächstmögliches Kraftwerk']
        if load_nodes_only:
            if self.load_case is None:
                self.set_load_case(LOAD_CASE)
                load_substations = self.load_nodes
                self.reset_loads()
            else:
                load_substations = self.load_nodes
            closest_ps = closest_ps[load_substations]
        return closest_ps
