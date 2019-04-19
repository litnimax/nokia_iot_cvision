import time


class Alarm(object):
    def __init__(self, history):
        print("Init alarm object...")
        self.alarms = []
        self.history = history

    def clear(self):
        self.alarms = []

    def update(self, zone_key):
        alarms_len = len(self.alarms)
        if (alarms_len > self.history):
            self.alarms.pop(0)
        alarm_event = {"key": zone_key, "epoch": int(time.time()), "type": "motion"}
        self.alarms.append(alarm_event)

    def get_alarms_list(self):
        return self.alarms

