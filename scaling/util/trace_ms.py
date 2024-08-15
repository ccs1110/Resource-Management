class Microservice_Trace(object):
    def __init__(self,name):
        self.name=name
        self.duration=0

    def clear_duration(self):
        self.duration=0
