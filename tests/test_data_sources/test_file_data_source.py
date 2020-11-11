import pytest
from dataskillet.data_sources import FileSystemDataSource
import modin.pandas as pd
import numpy as np

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
def data_source(csv_file, tmpdir):
    dir_path = csv_file.dirpath()
    ds = FileSystemDataSource.from_dir(metadata_dir=str(tmpdir), files_dir_path=dir_path)
    return ds


class TestFileSystemDataSource:
    def test_created_from_dir(self, csv_file):
        dir_path = csv_file.dirpath()
        ds = FileSystemDataSource.from_dir(metadata_dir=dir_path, files_dir_path=dir_path)
        assert ds.tables and len(ds.tables) == 1
        table = ds.tables['titanic']
        assert table.name == csv_file.purebasename
        assert pd.read_csv(csv_file).shape == table.dataframe.shape

    def test_add_from_file(self, csv_file):
        ds = FileSystemDataSource(metadata_dir=csv_file.dirpath())
        assert not ds.tables and len(ds.tables) == 0
        ds.add_table_from_file(str(csv_file))
        table = ds.tables['titanic']
        assert table.name == csv_file.purebasename
        assert pd.read_csv(csv_file).shape == table.dataframe.shape

    def test_file_preprocessing(self, csv_file):
        source_df = pd.read_csv(str(csv_file))
        df = source_df.copy()
        df = df.append([None]) # Empty row
        df['empty_column'] = None # Empty column
        df = df[list(source_df.columns)+['empty_column']]
        df = df.rename(columns={'ticket': 'Ticket Number '}) # Bad column name
        df.to_csv(csv_file, index=None)

        ds = FileSystemDataSource(metadata_dir=csv_file.dirpath())
        assert not ds.tables and len(ds.tables) == 0

        # No preprocessing changes nothing
        ds.add_table_from_file(str(csv_file), clean=False)
        assert ds.tables['titanic'].dataframe.shape == df.shape
        assert (ds.tables['titanic'].dataframe.dropna().values == df.dropna().values).all().all()
        ds.drop_table('titanic')
        assert not ds.tables

        # cleaning applied
        ds.add_table_from_file(str(csv_file), clean=True)
        # Empty rows removed, empty columns removed, columns renamed
        new_df = ds.tables['titanic'].dataframe
        empty_rows = pd.isnull(new_df).all(axis=1)
        assert not empty_rows.any()
        empty_columns = pd.isnull(new_df).all(axis=0)
        assert not empty_columns.any()
        assert len(new_df) == len(source_df) # Duplicate dropped
        assert new_df.columns[8] == 'ticket_number'

        preprocessing_dict = ds.tables['titanic'].preprocessing_dict
        assert preprocessing_dict['empty_rows'] == [9]
        assert preprocessing_dict['drop_columns'] == ['empty_column']
        assert preprocessing_dict['rename']['Ticket Number '] == 'ticket_number'

    def test_create_table(self, csv_file):
        ds = FileSystemDataSource(metadata_dir=csv_file.dirpath())
        assert not ds.tables and len(ds.tables) == 0
        sql = f"CREATE TABLE ('{str(csv_file)}', True)"
        query_result = ds.query(sql)
        assert query_result == 'OK'
        assert ds.tables and len(ds.tables) == 1
        table = ds.tables['titanic']
        assert table.name == csv_file.purebasename
        assert pd.read_csv(csv_file).shape == table.dataframe.shape

    def test_create_table_error_on_recreate(self, csv_file, data_source):
        assert data_source.tables['titanic']

        sql = f"CREATE TABLE ('{str(csv_file)}', True)"
        with pytest.raises(Exception):
            query_result = data_source.query(sql)

    def test_drop_table(self, data_source):
        assert data_source.tables['titanic']
        sql = f"DROP TABLE titanic"
        query_result = data_source.query(sql)
        assert query_result == 'OK'
        assert not data_source.tables and len(data_source.tables) == 0

    def test_select_column(self, csv_file, data_source):
        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM titanic"

        query_result = data_source.query(sql)

        assert query_result.name == 'passenger_id'

        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

    def test_select_all(self, csv_file, data_source):
        df = pd.read_csv(csv_file)
        sql = "SELECT * FROM titanic"
        query_result = data_source.query(sql)
        assert (query_result.columns == df.columns).all()
        values_left = df.values
        values_right = query_result.values
        assert values_left.shape == values_right.shape

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

    def test_select_limit_offset(self, csv_file, data_source):
        sql = "SELECT passenger_id FROM titanic LIMIT 2 OFFSET 2"
        query_result = data_source.query(sql)

        df = pd.read_csv(csv_file)['passenger_id']
        df = df.iloc[2:, :]
        df = df.iloc[:2, :]

        assert query_result.shape == df.shape
        assert (df.values == query_result.values).all().all()

    def test_select_multiple_columns(self, csv_file, data_source):
        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id, survived FROM titanic"

        query_result = data_source.query(sql)

        assert list(query_result.columns) == ['passenger_id', 'survived']

        values_left = df[['passenger_id', 'survived']].values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_select_const(self, csv_file, data_source):
        df = pd.read_csv(csv_file)
        df['const'] = 1

        sql = "SELECT passenger_id, 1 as const FROM titanic"

        query_result = data_source.query(sql)

        assert list(query_result.columns) == ['passenger_id', 'const']

        values_left = df[['passenger_id', 'const']].values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_select_operation(self, csv_file, data_source):
        df = pd.read_csv(csv_file)
        df['col_sum'] = df['passenger_id'] + df['survived']
        df['col_diff'] = df['passenger_id'] - df['survived']
        sql = "SELECT passenger_id + survived  as col_sum, passenger_id - survived as col_diff FROM titanic"
        query_result = data_source.query(sql)
        assert list(query_result.columns) == ['col_sum', 'col_diff']
        values_left = df[['col_sum', 'col_diff']].values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_select_where(self, csv_file, data_source):
        df = pd.read_csv(csv_file)
        out_df = df[df['survived'] == 1][['passenger_id', 'survived']]
        sql = "SELECT passenger_id, survived FROM titanic WHERE survived = 1"
        query_result = data_source.query(sql)
        assert list(query_result.columns) == ['passenger_id', 'survived']
        values_left = out_df[['passenger_id', 'survived']].values
        values_right = query_result.values
        assert values_left.shape == values_right.shape
        assert (values_left == values_right).all()

    def test_select_groupby_wrong_column(self, csv_file, data_source):
        sql = "SELECT survived, p_class, count(passenger_id) as count_passenger_id FROM titanic GROUP BY survived"
        with pytest.raises(Exception):
            query_result = data_source.query(sql)

    def test_select_aggregation_function_no_groupby(self, csv_file, data_source):
        df = pd.read_csv(csv_file)
        df = pd.DataFrame({'col_sum': [df['passenger_id'].sum()], 'col_avg': [df['passenger_id'].mean()]})
        sql = "SELECT sum(passenger_id) as col_sum, avg(passenger_id) as col_avg FROM titanic"
        query_result = data_source.query(sql)
        assert list(query_result.columns) == ['col_sum', 'col_avg']
        values_left = df[['col_sum', 'col_avg']].values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_groupby(self, csv_file, data_source):
        sql = "SELECT survived, p_class, count(passenger_id) as count_passenger_id FROM titanic GROUP BY survived, p_class HAVING survived = 1"
        query_result = data_source.query(sql)

        df = pd.read_csv(csv_file)
        df = df.groupby(['survived', 'p_class']).agg({'passenger_id': 'count'}).reset_index()
        df.columns = ['survived', 'p_class', 'count_passenger_id']
        df = df[df['survived'] == 1]

        assert (query_result.columns == df.columns).all()
        assert query_result.shape == df.shape

        assert (query_result.survived == 1).all()
        values_left = df.values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_inner_join(self, csv_file, data_source):
        df = pd.read_csv(csv_file)
        merge_df = pd.merge(df, df, how='inner', left_on=['passenger_id'], right_on=['p_class'])[['passenger_id_x', 'p_class_y']]
        merge_df.columns = ['passenger_id', 'p_class']
        sqls = ["SELECT passenger_id, p_class FROM titanic as t1 INNER JOIN titanic as t2 ON t1.passenger_id = t2.p_class",
                "SELECT passenger_id, p_class FROM titanic as t1 INNER JOIN titanic as t2 ON t2.p_class = t1.passenger_id"]
        for sql in sqls:
            query_result = data_source.query(sql)
            assert list(query_result.columns) == ['passenger_id', 'p_class']
            values_left = merge_df[['passenger_id', 'p_class']].values
            values_right = query_result.values
            assert (values_left == values_right).all().all()

    def test_inner_join_no_aliases(self, csv_file, tmpdir):
        p = tmpdir.join('titanic2.csv')
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

        dir_path = csv_file.dirpath()
        data_source = FileSystemDataSource.from_dir(metadata_dir=dir_path, files_dir_path=dir_path)
        assert len(data_source.tables) == 2

        df = pd.read_csv(csv_file)
        merge_df = pd.merge(df, df, how='inner', left_on=['passenger_id'], right_on=['p_class'])[['passenger_id_x', 'p_class_y']]
        merge_df.columns = ['passenger_id', 'p_class']
        sql = "SELECT passenger_id, p_class FROM titanic INNER JOIN titanic2 ON titanic.passenger_id = titanic2.p_class"
        query_result = data_source.query(sql)
        assert list(query_result.columns) == ['passenger_id', 'p_class']
        values_left = merge_df[['passenger_id', 'p_class']].values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_inner_join_col_access(self, csv_file, data_source):
        df = pd.read_csv(csv_file)
        merge_df = pd.merge(df, df, how='inner', left_on=['passenger_id'], right_on=['p_class'])[['passenger_id_x', 'p_class_y', 'sex_x']]
        merge_df.columns = ['passenger_id', 'p_class', 't1.sex']
        sql = "SELECT passenger_id, p_class, t1.sex FROM titanic as t1 INNER JOIN titanic as t2 ON t1.passenger_id = t2.p_class"
        query_result = data_source.query(sql)
        assert list(query_result.columns) == ['passenger_id', 'p_class', 't1.sex']
        values_left = merge_df.values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

        sql = "SELECT passenger_id, p_class, t1.sex as sex FROM titanic as t1 INNER JOIN titanic as t2 ON t1.passenger_id = t2.p_class"
        query_result = data_source.query(sql)
        assert list(query_result.columns) == ['passenger_id', 'p_class', 'sex']
        values_left = merge_df.values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_left_right_outer_joins(self, csv_file, data_source):
        df = pd.read_csv(csv_file)
        merge_df = pd.merge(df, df, how='left', left_on=['passenger_id'], right_on=['p_class'])[['passenger_id_x', 'p_class_y']]
        merge_df.columns = ['passenger_id', 'p_class']
        sql = "SELECT passenger_id, p_class FROM titanic as t1 LEFT JOIN titanic as t2 ON t1.passenger_id = t2.p_class"
        query_result = data_source.query(sql)
        assert merge_df.shape == query_result.shape
        assert list(query_result.columns) == ['passenger_id', 'p_class']
        values_left = merge_df.dropna().values
        values_right = query_result.dropna().values
        assert (values_left == values_right).all().all()

        merge_df = pd.merge(df, df, how='right', left_on=['passenger_id'], right_on=['p_class'])[
            ['passenger_id_x', 'p_class_y']]
        merge_df.columns = ['passenger_id', 'p_class']
        sql = "SELECT passenger_id, p_class FROM titanic as t1 RIGHT JOIN titanic as t2 ON t1.passenger_id = t2.p_class"
        query_result = data_source.query(sql)
        assert merge_df.shape == query_result.shape
        assert list(query_result.columns) == ['passenger_id', 'p_class']
        values_left = merge_df.dropna().values
        values_right = query_result.dropna().values
        assert (values_left == values_right).all().all()

        merge_df = pd.merge(df, df, how='outer', left_on=['passenger_id'], right_on=['p_class'])[
            ['passenger_id_x', 'p_class_y']]
        merge_df.columns = ['passenger_id', 'p_class']
        sql = "SELECT passenger_id, p_class FROM titanic as t1 FULL JOIN titanic as t2 ON t1.passenger_id = t2.p_class"
        query_result = data_source.query(sql)
        assert merge_df.shape == query_result.shape
        assert list(query_result.columns) == ['passenger_id', 'p_class']
        values_left = merge_df.dropna().values
        values_right = query_result.dropna().values
        assert (values_left == values_right).all().all()

    def test_subquery_simple(self, csv_file, data_source):
        sql = "SELECT * FROM (SELECT * FROM titanic) as t1"
        query_result = data_source.query(sql)
        df = pd.read_csv(csv_file)

        assert query_result.shape == df.shape
        values_left = df.dropna().values
        values_right = query_result.dropna().values
        assert (values_left == values_right).all()

    def test_subquery_groupby(self, csv_file, data_source):
        sql = "SELECT survived, p_class, count(passenger_id) as count FROM (SELECT * FROM titanic WHERE survived = 1) as t1 GROUP BY survived, p_class"
        query_result = data_source.query(sql)

        df = pd.read_csv(csv_file)
        df = df[df.survived == 1]
        df = df.groupby(['survived', 'p_class']).agg({'passenger_id': 'count'}).reset_index()

        assert query_result.shape == df.shape
        values_left = df.dropna().values
        values_right = query_result.dropna().values
        assert (values_left == values_right).all()

    def test_subquery_where(self, csv_file, data_source):
        sql = "SELECT survived, p_class, passenger_id FROM titanic WHERE passenger_id IN (SELECT passenger_id FROM titanic WHERE survived = 1)"
        query_result = data_source.query(sql)

        df = pd.read_csv(csv_file)
        df = df[df.survived == 1]
        df = df[['survived', 'p_class', 'passenger_id']]

        assert query_result.shape == df.shape
        values_left = df.dropna().values
        values_right = query_result.dropna().values
        assert (values_left == values_right).all()

    def test_subquery_select(self, csv_file, data_source):
        sql = "SELECT survived, (SELECT passenger_id FROM titanic LIMIT 1) as pid FROM titanic"
        query_result = data_source.query(sql)
        assert (query_result['pid'] == 1).all()

    def test_show_tables(self, csv_file, data_source):
        sql = "SHOW TABLES"
        query_result = data_source.query(sql)
        assert (query_result.values == np.array([['titanic', str(csv_file)]])).all()

        data_source.drop_table('titanic')
        query_result = data_source.query(sql)
        assert query_result.empty
