class Logger():
    def __init__(self, active):
        self.active = active

    def log(self, msg):
        if self.active:
            print(msg, flush=True)