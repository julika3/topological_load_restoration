import json


def save_as_json(received_df, distribution_distances, solution_dict, scenario_parameters,
                 filepath=None):
    datetime_identifier = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    data_new = serialised_results(received_df, distribution_distances, solution_dict, scenario_parameters)

    filepath = '' if filepath is None else filepath

    try:
        with open(filepath) as f:
            data = json.load(f)
            data = json.loads(data)
        data.update({datetime_identifier: data_new})
        with open(filepath, 'w') as f:
            json.dump(json.dumps(data), f)
        print(f'Appended to file {filepath}.')

    except FileNotFoundError:
        data = {datetime_identifier: data_new}
        with open(filepath, 'x') as f:
            json.dump(json.dumps(data), f)
        print(f'New file named {filepath} created.')
