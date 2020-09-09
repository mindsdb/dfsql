class Statement:
    def __init__(self, raw=None):
        self.raw = raw

    def __str__(self):
        return self.raw
