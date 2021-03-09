import pandas as pd
import pytest

from pdsql.exceptions import QueryExecutionException, pdsqlException


class TestQuickInterface:
    def test_simple_select(self, csv_file):
        from pdsql.extensions import sql_query

        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM whatever_table"

        query_result = sql_query(sql, from_tables={'whatever_table': df})
        assert query_result.name == 'passenger_id'
        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

        # Run query again to ensure that everything was cleaned up properly
        query_result = sql_query(sql, from_tables={'whatever_table': df})
        assert query_result.name == 'passenger_id'
        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

    def test_select_join(self, csv_file):
        from pdsql.extensions import sql_query
        df = pd.read_csv(csv_file)
        merge_df = pd.merge(df, df, how='inner', left_on=['passenger_id'], right_on=['p_class'])[
            ['passenger_id_x', 'p_class_y']]
        merge_df.columns = ['passenger_id', 'p_class']

        # Use one table for self join
        sql = "SELECT passenger_id, p_class FROM titanic as t1 INNER JOIN titanic as t2 ON t1.passenger_id = t2.p_class"
        query_result = sql_query(sql, from_tables={'titanic': df})

        assert list(query_result.columns) == ['passenger_id', 'p_class']
        values_left = merge_df[['passenger_id', 'p_class']].values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

        # Use two separate tables
        sql = "SELECT passenger_id, p_class FROM t1 INNER JOIN t2 ON t1.passenger_id = t2.p_class"
        query_result = sql_query(sql, from_tables={'t1': df, 't2': df})

        assert list(query_result.columns) == ['passenger_id', 'p_class']
        values_left = merge_df[['passenger_id', 'p_class']].values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_error_table_not_found(self, csv_file):
        from pdsql.extensions import sql_query
        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM whatever_table INNER JOIN missing_table ON id"
        with pytest.raises(QueryExecutionException):
            sql_query(sql, from_tables={'whatever_table': df})

    def test_error_wrong_table_name(self, csv_file):
        from pdsql.extensions import sql_query

        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM whatever_table"

        with pytest.raises(pdsqlException):
            sql_query(sql, from_tables={'wrong_table': df})

        # Run again to make sure it works after a failure
        query_result = sql_query(sql, from_tables={'whatever_table': df})
        assert query_result.name == 'passenger_id'
        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

    def test_error_no_tables(self):
        from pdsql.extensions import sql_query
        sql = "SELECT passenger_id FROM whatever_table"

        with pytest.raises(pdsqlException):
            sql_query(sql, None)

        with pytest.raises(pdsqlException):
            sql_query(sql, from_tables={})

        with pytest.raises(pdsqlException):
            sql_query(sql, from_tables=[])

    def test_error_extra_tables(self, csv_file):
        from pdsql.extensions import sql_query
        df = pd.read_csv(csv_file)
        sql = "SELECT passenger_id FROM whatever_table"

        with pytest.raises(pdsqlException):
            sql_query(sql, from_tables={'whatever_table': df, 'another_table': df})

    def test_custom_functions(self, csv_file):
        from pdsql.extensions import sql_query
        df = pd.read_csv(csv_file)
        sql = "SELECT sex, mode(survived) as mode_survived FROM titanic GROUP BY sex"

        func = lambda x: x.value_counts(dropna=False).index[0]

        query_result = sql_query(sql, from_tables={'titanic': df}, custom_functions={'mode': func})

        df = df.groupby(['sex']).agg({'survived': func}).reset_index()
        df.columns = ['sex', 'mode_survived']

        assert (query_result.columns == df.columns).all()
        assert query_result.shape == df.shape

        values_left = df.values
        values_right = query_result.values
        assert (values_left == values_right).all().all()
