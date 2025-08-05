# specify network name here
NETWORK_MODEL_NAME = 'example'

NETWORK_MODELS = {'template': {'filepath_network': 'network_model_template.xlsx',
                               'filepath_substation_infos': 'substation_infos_template.xlsx'},
                  'example': {'filepath_network': 'example_network.xlsx',
                               'filepath_substation_infos': 'example_substation_infos.xlsx'}}


LOAD_CASE = 'load_case_1'  # 'load_case_2'
LOAD_NAMES = {'load_case_1': 'Load Name 1',
              'load_case_2': 'Load Name 2'}

## column names
# network model
NETWORK_ELEMENT = "NE_name"
SUBSTATION_NAME_1 = "substation_1_name"
SUBSTATION_NAME_2 = "substation_2_name"

## substation infos
POWER_COLUMN = 'generation_power'
TYPE_COLUMN = 'type'  # template allows for the specification of a substation type
NODE_TYPE_FILTER = None  # enables filtering of nodes if network model i.e. includes nodes for wind parks
SUBSTATION_COLUMN = 'substation'

# scenarios
SCENARIO_NAME = 'scenario_name'
SCENARIO_LOCATION = 'substation'
SCENARIO_GENERATION_VALUE = 'generation_value'
SCENARIO_DESCRIPTION = 'description'


# meta data for saving
SCENARIO_PARAMETERS = {'network_model_name': NETWORK_MODEL_NAME,
                       'load_variant': LOAD_CASE,
                       'kwargs': {'scenario_descriptor': 'Basisszenario'}}
SOLUTION_DICT = {}
RESULT_FILEPATH = 'simulation_results.json'


# assign colors to the power stations
ASSIGNED_COLORS = {'KW1': '#33BECC',
                   'KW2': '#0074A8',
                   'A': '#D56231',
                   'B': '#CFBB7B',
                   'C': '#B5BAB6',
                   'D': '#77202E',
                   'E': '#4B6D41',
                   'F': '#6F6C70'}


# in MW --> this achieves the same total load (for 60 nodes) as the Mean load estimate for 50Hertz substations
LOAD_ESTIMATE = 175

# Standard power for a new built gas power station
STANDARD_POWER = 500

# optimisation
SLACK = 0.95

### constants determining execution behaviour
PRINT_STATEMENTS = True  # switches printing to console on or off
SAVE = False             # if set to True all results will be saved to JSON files per default,
                         # changes can also be made individually for functions
SHOW_FIGS = True         # shows all figs after creation
