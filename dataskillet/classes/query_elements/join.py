class Join:

    def __init__(self, type = None, table = None, on = None):
        self.table = table
        self.type = type
        self.on = on

    @staticmethod
    def parse_string(str):
        pass