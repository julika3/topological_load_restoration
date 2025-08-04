# specify network name here
NETWORK_MODEL_NAME = ''

NETWORK_MODELS = {NETWORK_MODEL_NAME: {'filepath_network': 'network_model.xlsx',
                                       'filepath_substation_infos': 'substation_infos.xlsx'},
                  }


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


LOAD_ESTIMATE = 175  # MW --> this achieves the same total load as the Mean load estimate for 50Hertz substations

# optimisation
OPTIMISATION_PRECISION = 1
SLACK = 0.95
LOAD_CASE = 'load_case_1'  # 'load_case_2'

# meta data for saving
SCENARIO_PARAMETERS = {'network_model_name': NETWORK_MODEL_NAME,
                       'load_variant': LOAD_CASE,
                       'kwargs': {'scenario_descriptor': 'Basisszenario'}}
SOLUTION_DICT = {}
RESULT_FILEPATH = 'simulation_results.json'

ASSIGNED_COLORS = {'Boxberg': '#33BECC',
                   'Jaenschwalde': '#0074A8',
                   'Lippendorf': '#D56231',
                   'Reuter-West': '#CFBB7B',
                   'Rostock': '#B5BAB6',
                   'Schkopau': '#77202E',
                   'Schwarze Pumpe': '#4B6D41',
                   'Thyrow': '#6F6C70'}

LOAD_NAMES = {'load_case_1': 'Load Name 1',
              'load_case_2': 'Load Name 2'}

STANDARD_POWER = 500


### constants determining execution behaviour
PRINT_STATEMENTS = True  # switches printing to console on or off
SAVE = False             # if set to True all results will be saved to JSON files per default,
                         # changes can also be made individually for functions
SHOW_FIGS = True         # shows all figs after creation
