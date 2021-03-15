from dfsql.config import Configuration

pd = None
if Configuration.USE_MODIN:
    import modin.pandas as pd
else:
    import pandas as pd
