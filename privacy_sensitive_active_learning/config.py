import yaml

CONFIG_FILE = 'config_file.yaml'

def load():
    with open(CONFIG_FILE, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config