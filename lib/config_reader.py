import json


class ConfigReader():
    def __init__(self, filename):
        self.config = json.loads(filename)

    def get_config(self, name):
        return self.config[name]
