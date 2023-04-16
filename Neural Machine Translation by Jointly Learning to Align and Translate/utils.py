import yaml

def write_yaml(params_filename, dictionary):
    try:
        with open(params_filename, 'w') as f:
            yaml.safe_dump(dictionary, f)
        return "parameters successfully dumped"
    except Exception as e:
        return e.message

def read_yaml(params_filename):
    with open(params_filename, 'r') as f:
        return yaml.safe_load(f)
