# topological_load_restoration
main.py:
- create a RestorationNetwork class object for your network and run the executables
- filepaths and model name must previously be specified in constants.py
- an example network was included in the data folder; the size of the example network makes meaningful insights unlikely

constants.py:
- specifies the filenames for the network model and additional substation information plus header names in said files
- parameters like the slack or the standard power for new power stations can be changed here
- additional parameters to determine whether fidures are shown, results are saved and messages printed to the console

executables.py:
- all applications of the tool (ie load restoration, or scenario application and analysis) are compiled into individual functions
- these may be executed by keeping all default parameter values and only passing a RestorationNetwork class object

RestorationNetwork Class:
- this class builds a network model and combines all class attributes and class functions required to use the functionalities in the analysis folder

analysis includes:
- resilience_indicators (calculation of all Resilienzindikatoren as detailed in thesis)
- load_restoration_strategy_functions (functions for the Superpositionsstrategie and Resilienzindikatorenstrategie as well as functions to analyse the results of simulations)
- optimisation_strategy_functions (function of the regular Optimierungsstrategie, and the identification of the optimal location for a power station build)

additional_finctionalities includes:
- ValidationError.py (raised if load_restoration results are implausible)
- helper_functions.py (functions for saving and retrieving data; conditional printing to console)
- visualisation.py (all result visualisation functions)

data
- include your network_model and substation_info files here
