import json
from datetime import datetime

import numpy as np
import pandas as pd
from _plotly_utils.colors import qualitative

from constants import PRINT_STATEMENTS, RESULT_FILEPATH


def print_me(print_statement):
    if PRINT_STATEMENTS:
        print(print_statement)


def result_table_from_scenarios(identifier_list, path=RESULT_FILEPATH, filename=None):
    info_pairs = {'load_factor': 'Versorgungsgrad',
                  'total_path_length': 'Gesamte Pfadlänge',
                  'total_path_deviation': 'Gesamte Pfadabweichung',
                  'mean_path_length': 'Mittlere Pfadlänge',
                  'mean_path_deviation': 'Mittlere Pfadabweichung'}
    res = pd.DataFrame(columns=identifier_list, index=info_pairs.values())
    for identifier in identifier_list:
        data = json_to_dict(path, identifier)
        vals = [data['solution'][x] for x in info_pairs.keys()]
        res[identifier] = vals

    if filename is not None:
        res.round(4).to_excel(f'{filename}.xlsx')
    return res


if __name__ == '__main__':
    #res = result_table_from_scenarios(['base_scenario_superposition', 'base_scenario_resilienzindikatoren',
    #                                   'base_scenario_optimisation'])
    #res.round(4).to_excel('Vergleich_der_VWA-Strategien_im_Basisszenario.xlsx')

    id_list = [f'Abschaltung_Rostock_{x}' for x in ['superposition', 'resilienzindikatoren', 'optimierung']]
    filename = f'Vergleich_der_VWA-Strategien_Abschaltung_Rostock.xlsx'
    result_table_from_scenarios(id_list, filename=filename)


def generate_color_dict(ps_list):
    color_list = qualitative.Alphabet + qualitative.Light24
    color_dict = dict([(ps, color_list[n]) for (n, ps) in enumerate(ps_list)])
    return color_dict


def save_as_json(dispatched_power_df, shortest_distance_matrix, solution_dict, scenario_parameters,
                 filepath=RESULT_FILEPATH, dataset_identifier=None):
    """
    stores the results of a simulation in the json file specified in filepath
    this allows data being retrieved for visualisations and analysis without requiring a recalculation everytime
    :param dispatched_power_df: dataframe that specifies which power station dispatches how much power to which node
    :param shortest_distance_matrix: shortest path lengths between load nodes and power stations
    :param solution_dict: dict entailing the keyword value pairs 'load_factor', 'total_path_length',
            'total_path_deviation': total_path_deviation, 'mean_path_length', 'mean_path_deviation'
    :param scenario_parameters: includes information on the analysed scenario 'network_model_name', 'load_variant',
                                'strategy', 'kwargs': {'scenario_descriptor'}
    :param filepath: specify the storage filepath otherwise the filepath from constants is used; if the specified file
                    doesn't exist a new one is created
    :param dataset_identifier:
    :return:
    """
    dataset_identifier = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") if dataset_identifier is None \
        else dataset_identifier
    solution_dict.update({'max_path_length': shortest_distance_matrix[dispatched_power_df > 1].max().max(),  # .to_json(),
                          'shortest_distance_matrix': shortest_distance_matrix.to_dict('index'),
                          'dispatched_power': dispatched_power_df.to_dict('index')
                          })
    data_new = {'scenario_parameters': scenario_parameters,
                'solution': solution_dict}

    filepath = '' if filepath is None else filepath

    try:
        with open(filepath) as f:
            data = json.load(f)
            data = json.loads(data)
        data.update({dataset_identifier: data_new})
        with open(filepath, 'w') as f:
            json.dump(json.dumps(data), f)
        print(f'Appended to file {filepath}.')

    except FileNotFoundError:
        data = {dataset_identifier: data_new}
        with open(filepath, 'x') as f:
            json.dump(json.dumps(data), f)
        print(f'New file named {filepath} created.')


def save_figure(fig, fig_title, save_as):
    """
    save figures to a file in result folder figs return 0 if save_as is False
    :param fig: go.Figure object to be saved
    :param fig_title: filename
    :param save_as: allow specification of file type (PDF/HTML/JSON) but default to JSON
    :return:
    """
    if not save_as:
        return 0
    if save_as == 'PDF':
        try:
            fig.write_image(f"figs/{fig_title}.pdf")
        except ValueError:
            print('PDF Exports werden durch eine Gruppenrichtlinie geblockt')
    elif save_as == 'HTML':
        fig.write_html(f'figs/{fig_title}.html')
    else:
        save_figures_to_file([fig], f'figs/{fig_title}.json')


def save_figures_to_file(figures, filename):
    """
    Speichert eine Liste von plotly.graph_objects.Figure Objekten in einer Datei im JSON-Format.

    Parameters:
    - figures (list of plotly.graph_objects.Figure): Liste der Figure Objekte, die gespeichert werden sollen.
    - filename (str): Der Name der Datei, in der die Figure Objekte gespeichert werden sollen.
    """
    def default(obj):
        """
        Hilfsfunktion zur Handhabung von nicht serialisierbaren Objekten.
        Wandelt numpy.ndarray in Listen um.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    figures_json = [fig.to_plotly_json() for fig in figures]
    with open(filename, 'w') as f:
        json.dump(figures_json, f, default=default)
    print(f"{len(figures)} Figures wurden erfolgreich in {filename} gespeichert.")


def json_to_dict(filepath=RESULT_FILEPATH, dataset_identifier=None):
    with open(filepath) as f:
        data = json.load(f)
        data = json.loads(data)

    iterator = data.keys() if dataset_identifier is None else [dataset_identifier]

    for i in iterator:
        data[i]['solution']['shortest_distance_matrix'] = pd.DataFrame.from_dict(data[i]['solution']['shortest_distance_matrix'],
                                                                     orient='index')
        data[i]['solution']['dispatched_power'] = pd.DataFrame.from_dict(data[i]['solution']['dispatched_power'],
                                                                          orient='index')

    if dataset_identifier is not None:
        return data[dataset_identifier]
    else:
        return data


def dataframe_to_file(df, full_title=None, keyword=None):
    if full_title is None:
        datestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        full_title = f'{keyword + "_" if keyword is not None else ""}{datestamp}'
    final_filepath = f'results/{full_title}.xlsx'
    df.to_excel(final_filepath)
    print(f"Dataframe wurden erfolgreich in {final_filepath} gespeichert.")


def retrieve_relevant_comparison_data(id_string, filepath=RESULT_FILEPATH):
    data = json_to_dict(filepath, id_string)
    result = data['solution']['dispatched_power']
    paths = data['solution']['shortest_distance_matrix']
    scenario = data['scenario_parameters']['kwargs']['scenario_descriptor']
    return result, paths, scenario
