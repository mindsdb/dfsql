# TODO: Allow this to be a configuration
import os
import ray
from dataskillet import CONFIG



pd = None


class Table:

    def __init__(self):

        global pd

        # Initialize the dataframe engine if not yet initialized
        if pd is None:
            # set modin's config variables
            os.environ["MODIN_ENGINE"] = CONFIG.MODIN_ENGINE  # Modin will use Ray
            os.environ["MODIN_OUT_OF_CORE"] = CONFIG.MODIN_OUT_OF_CORE
            os.environ["MODIN_MEMORY"] = CONFIG.MODIN_MEMORY
            import modin.pandas as pd_f
            pd = pd_f

        self.df = None #type: pandas.DataFrame
        self.metadata = None


    def read_csv(self, from_file = None):
        """
        Generic method to load the data into the Table

        :param from_file: file path to load from, must be pandas compatible
        :return:
        """
        self.df = pd.read_csv   (from_file)


    def analyze(self):
        """
        This will analyze all columns and store the analysis results in self.metadata

        :return: self.df_metadata
        """
        pass




    def splits(self, percentages):
        pass





