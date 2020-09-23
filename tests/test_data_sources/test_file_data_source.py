import pytest
from dataskillet.data_sources import FileSystemDataSource
from pandas.testing import assert_frame_equal
import modin.pandas as pd
import pandas


@pytest.fixture()
def csv_file(tmpdir):
    # Titanic dataset first 10 lines of train
    p = tmpdir.join('titanic.csv')
    content = """passenger_id,survived,p_class,name,sex,age,sib_sp,parch,ticket,fare,cabin,embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S
    """
    p.write_text(content, encoding='utf-8')
    return p


@pytest.fixture()
def data_source(csv_file):
    dir_path = csv_file.dirpath()
    ds = FileSystemDataSource.from_dir(dir_path)
    return ds


class TestFileSystemDataSource:
    def test_created_from_dir(self, csv_file):
        dir_path = csv_file.dirpath()
        ds = FileSystemDataSource.from_dir(dir_path)

        assert ds.tables and len(ds.tables) == 1

        table = ds.tables['titanic']
        assert table.name == csv_file.purebasename

        assert pd.read_csv(csv_file).shape == table.df.shape

    def test_select_column(self, csv_file, data_source):
        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM titanic"

        query_result = data_source.query(sql)

        assert query_result.name == 'passenger_id'

        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

    def test_select_column_alias(self, csv_file, data_source):
        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id as p1 FROM titanic"

        query_result = data_source.query(sql)

        assert query_result.name == 'p1'

        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

    def test_select_distinct(self, csv_file, data_source):
        sql = "SELECT DISTINCT survived FROM titanic"
        query_result = data_source.query(sql)
        assert query_result.name == 'survived'
        assert list(query_result.values) == [0, 1]

