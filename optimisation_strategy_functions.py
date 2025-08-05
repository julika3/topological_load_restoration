from datetime import datetime

import cvxpy as cp
import pandas as pd

from constants import SLACK, SCENARIO_NAME, SCENARIO_LOCATION, SCENARIO_GENERATION_VALUE, \
    SCENARIO_DESCRIPTION
from RestorationNetwork import RestorationNetwork
from helper_functions import print_me
from load_restoration_strategy_functions import validate_result, evaluate_restoration_result


class Node(object):
    """ A node with power supply (power station) or power demand (load) """
    def __init__(self, power=0, name='', load_factor=1):
        self.power = power
        self.name = name
        self.load_factor = load_factor
        self.edge_flows = []  # turn into edge flow regardless of path
        self.distances = []

    # The nodes constraints (as per the optimisation problem) are dependent on them being either
    # a power station: sum of flows going out of the power station cannot exceed the inherent generation power
    # or load node: sum of flows must be greater than the allocated demand minus the allowed slack and
    #               less than the allocated demand plus slack and less than the actual demand
    def constraints(self):
        if self.power > 0:  # load
            return [sum(f for f in self.edge_flows) >= SLACK * self.power,
                    sum(f for f in self.edge_flows) <= min(self.power / self.load_factor, (2 - SLACK) * self.power)]
        else:  # generator
            # not all energy must be distributed (if more generation than load is in network)
            return [(sum(f for f in self.edge_flows)) >= self.power if self.load_factor > 1 else
                    (sum(f for f in self.edge_flows)) == self.power]


class Edge(object):
    """A bidirectional Edge connecting two nodes"""
    def connect(self, in_node, out_node):
        """Connects two nodes via the edge and appends the nodes edge_flows accordingly"""
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.flow)
        in_node.distances.append(self.length)

    def __init__(self, length, in_node: Node, out_node: Node):
        self.flow = cp.Variable()
        self.in_name = in_node.name
        self.out_name = out_node.name
        self.is_active = cp.Variable(boolean=True)
        self.length = length
        self.max_flow = in_node.power / in_node.load_factor

        self.connect(in_node, out_node)

    # the objective function relevant price of an edge is determined by the path length it represents
    # the cost is zero if the edge is not active and therefore no flow is transported via this edge
    def cost(self):
        return self.length * self.is_active

    # edge flows must be less than zero making them effectively unidirectional
    # limiting the edge flows to being >= - max_load * self.is_active constraints them to being zero
    # if is_active is False (binary interpretation of the bool value);
    # setting max_load as an upper boundary for flows helps with computation time
    def constraints(self, max_load):
        return [self.flow <= 0,
                self.flow >= - max_load * self.is_active]


def create_optimisation_problem(restoration_network: RestorationNetwork):
    """
    reinterprets the network graph of the restoration_network as an optimisation problem
    uses the shortest distance matrix (path_lengths_df) as a basis
    returns: a cvxpy optimisation model and a dictionary of the corresponding node objects (name str as keys)
             as well as a list of the edge objects
    """
    # manipulate the network first to split nodes containing both generation power and load demand into
    # two separate nodes with otherwise unchanged properties
    restoration_network.handle_dual_substations()
    substations = restoration_network.load_series_manipulated[restoration_network.load_series_manipulated > 0].index
    power_stations = restoration_network.generators
    max_load = restoration_network.load_series.max()

    nodes_dict = {}
    edges = []

    for ss in substations:
        nodes_dict.update({ss: Node(restoration_network.load_series_manipulated[ss], ss,
                                    restoration_network.load_factor)})
    for ps in power_stations:
        nodes_dict.update({ps: Node(- restoration_network.generation_series[ps], ps)})
    for ss in substations:
        for ps in power_stations:
            edges.append(Edge(restoration_network.sdm_manipulated.loc[ps, ss], nodes_dict[ss], nodes_dict[ps]))
    constraints = []
    for obj in nodes_dict.values():
        constraints += obj.constraints()
    for obj in edges:
        constraints += obj.constraints(max_load)
    problem = cp.Problem(cp.Minimize(cp.sum([edge.cost() for edge in edges])), constraints)
    return problem, nodes_dict, edges


def get_detailed_result(restoration_network: RestorationNetwork, optimised_edges):
    """
    takes the list of edges after the optimisation was performed and interprets their flow values
    in order to gather the dispatched power
    :param restoration_network:
    :param optimised_edges: list of class Edge objects
    :return: dataframe detailing the dispatched power
    """
    received_dict = {x: {'sum': 0} for x in restoration_network.load_series_manipulated.index}
    for edge in optimised_edges:
        received_dict[edge.in_name]['sum'] = received_dict[edge.in_name]['sum'] + edge.flow.value
        received_dict[edge.in_name].update({edge.out_name: edge.flow.value})
    dispatched_power = pd.DataFrame(received_dict).fillna(0)
    if len(restoration_network.dual_substations) > 0:
        dispatched_power.rename({f'{ss}_load': ss for ss in restoration_network.dual_substations}, axis=1, inplace=True)
    available_keys = sorted(dispatched_power.index.intersection(restoration_network.generators))
    dispatched_power = - dispatched_power.loc[available_keys]

    validate_result(dispatched_power, restoration_network.generation_series)

    return dispatched_power


def perform_optimisation(restoration_network, solution_dict=None):
    """
    creates, solves and summarises the optimisation problem for the given RestorationNetwork
    :param restoration_network:
    :param solution_dict: if given the analysed results will be saved in this dictionary
    :return: dataframe detailing the dispatched power
    """
    prob, nodes_dict, edges = create_optimisation_problem(restoration_network)
    solve_starttime = datetime.now()
    print_me(f'Starting optimisation @ {solve_starttime}')
    result = prob.solve(solver='HIGHS')
    solve_endtime = datetime.now()
    print_me(f'Solving time: {solve_endtime - solve_starttime}')
    dispatched_power = get_detailed_result(restoration_network, edges)

    if solution_dict is not None:
        solution_dict.update({'total_path_length': float(result),
                              'solving_time': str((solve_endtime - solve_starttime).seconds)
                              })

    return dispatched_power


def find_optimal_new_location(restoration_network: RestorationNetwork):
    dispatched_power_df = pd.DataFrame()
    optimal_location = ''
    optimal_value = None

    for ss in restoration_network.substations:
        scenario_dict = {SCENARIO_NAME: f'Neubau_{ss}',
                         SCENARIO_LOCATION: ss,
                         SCENARIO_GENERATION_VALUE: 500,
                         SCENARIO_DESCRIPTION: f'Neues Kraftwerk {ss}'}
        restoration_network.apply_scenario(scenario_dict, overwrite_prev=True)

        dispatched_power_df = perform_optimisation(restoration_network)

        mean_path, mean_path_deviation = evaluate_restoration_result(restoration_network,
                                                                     dispatched_power_df, mean=True)
        if (optimal_value is None) or (optimal_value > mean_path):
            optimal_value = mean_path
            optimal_location = ss

    return optimal_location, dispatched_power_df
