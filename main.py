from executables import *

if __name__ == "__main__":
    print('Starting Programme')
    print('Create Network')
    my_restoration_network = RestorationNetwork('50Hertz')
    my_restoration_network.set_load_case(LOAD_CASE, scale=True)
    run_superposition_restoration(my_restoration_network)
    run_resilience_indicator_restoration(my_restoration_network)
    run_scenario(my_restoration_network, build=True, node='Hamburg-Ost')