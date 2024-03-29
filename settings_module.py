import json

class Settings():
    def __init__(self, settings_file):
        self.settings_file = settings_file
        self.settings = self.read_settings_from_file()

    def __del__(self): #не сохраняет
        print("Save settings...")
        self.write_settings_to_file()

    def read_settings_from_file(self):
        try:
            with open(self.settings_file) as file:
                return json.load(file)
        except FileNotFoundError:
            return {'areas': {}, 'threshold': 5, 'width': 640, 'height': 360, 'min_area': 100}

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

    def get_size(self, reverse=False):
        if reverse:
            return self.settings['width'], self.settings['height']
        else:
            return self.settings['height'], self.settings['width']

    def set_min_area(self, min_area):
        self.settings['min_area'] = min_area
        self.write_settings_to_file()

    def get_min_area(self):
        return self.settings['min_area']

