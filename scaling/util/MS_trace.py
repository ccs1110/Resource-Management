class MS_trace(object):
    def __init__(self,name):
        self.name=name
        self.duration=0
        self.weight=0

    def clear_duration(self):
        self.duration=0

