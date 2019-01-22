import json

class Settings():
    def __init__(self, settings_file):
        self.settings_file = settings_file
        self.settings = self.read_settings_from_file()

    def read_settings_from_file(self):
        try:
            with open(self.settings_file) as file:
                return json.load(file)
        except FileNotFoundError:
            return {'areas': {}, 'threshold': 5, 'width': 640, 'height': 360}

    def write_settings_to_file(self):
        with open(self.settings_file, 'w') as outfile:
            json.dump(self.settings, outfile)

    def set_areas(self, areas):
        self.settings['areas'] = areas
        self.write_settings_to_file()

    def get_areas(self):
        return self.settings['areas']

    def set_threshold(self, threshold):
        if threshold % 2 == 0:
            threshold += 1
        self.settings['threshold'] = threshold
        self.write_settings_to_file()

    def get_threshold(self):
        return self.settings['threshold']

    def set_size(self, width, height):
        self.settings['width'] = width
        self.settings['height'] = height
        self.write_settings_to_file()

    def get_size(self):
        return self.settings['width'], self.settings['height']

