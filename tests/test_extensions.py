import modin.pandas as modin_pd
import pandas as pd
import pytest

from dataskillet.exceptions import QueryExecutionException, DataskilletException


class TestQuickInterface:
    def test_simple_select(self, csv_file):
        from dataskillet.extensions import sql_query

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
        from dataskillet.extensions import sql_query
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
        from dataskillet.extensions import sql_query
        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM whatever_table INNER JOIN missing_table ON id"
        with pytest.raises(QueryExecutionException):
            sql_query(sql, from_tables={'whatever_table': df})

    def test_error_wrong_table_name(self, csv_file):
        from dataskillet.extensions import sql_query

        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM whatever_table"

        with pytest.raises(DataskilletException):
            sql_query(sql, from_tables={'wrong_table': df})

        # Run again to make sure it works after a failure
        query_result = sql_query(sql, from_tables={'whatever_table': df})
        assert query_result.name == 'passenger_id'
        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

    def test_error_no_tables(self):
        from dataskillet.extensions import sql_query
        sql = "SELECT passenger_id FROM whatever_table"

        with pytest.raises(DataskilletException):
            sql_query(sql, None)

        with pytest.raises(DataskilletException):
            sql_query(sql, from_tables={})

        with pytest.raises(DataskilletException):
            sql_query(sql, from_tables=[])

    def test_error_extra_tables(self, csv_file):
        from dataskillet.extensions import sql_query
        df = pd.read_csv(csv_file)
        sql = "SELECT passenger_id FROM whatever_table"

        with pytest.raises(DataskilletException):
            sql_query(sql, from_tables={'whatever_table': df, 'another_table': df})


class TestPandasExtension:
    def test_df_sql_simple_select(self, csv_file):
        import dataskillet.extensions

        df = pd.read_csv(csv_file)

        sql_queries = [
            "SELECT passenger_id FROM temp",
            "SELECT passenger_id",
        ]
        for sql in sql_queries:
            query_result = df.sql(sql)
            assert query_result.name == 'passenger_id'

            values_left = df['passenger_id'].values
            values_right = query_result.values
            assert (values_left == values_right).all()

    def test_df_sql_nested_select_in(self, csv_file):
        import dataskillet.extensions

        df = pd.read_csv(csv_file)

        sql_queries = [
            "SELECT survived, p_class, passenger_id WHERE passenger_id IN (SELECT passenger_id WHERE survived = 1)",
            "SELECT survived, p_class, passenger_id FROM temp WHERE passenger_id IN (SELECT passenger_id WHERE survived = 1)",
            "SELECT survived, p_class, passenger_id WHERE passenger_id IN (SELECT passenger_id FROM temp WHERE survived = 1)",
            "SELECT survived, p_class, passenger_id FROM temp WHERE passenger_id IN (SELECT passenger_id FROM temp WHERE survived = 1)"
        ]

        for sql in sql_queries:
            query_result = df.sql(sql)

            expected_df = df[df.survived == 1][['survived', 'p_class', 'passenger_id']]

            assert query_result.shape == expected_df.shape
            values_left = expected_df.dropna().values
            values_right = query_result.dropna().values
            assert (values_left == values_right).all()

    def test_df_sql_nested_select_from(self, csv_file):
        import dataskillet.extensions

        df = pd.read_csv(csv_file)[['passenger_id', 'fare']]
        sql_queries = [
            "SELECT * FROM (SELECT passenger_id, fare FROM temp) as t1",
            "SELECT * FROM (SELECT passenger_id, fare) as t1",
        ]

        for sql in sql_queries:
            query_result = df.sql(sql)

            assert query_result.shape == df.shape
            values_left = df.dropna().values
            values_right = query_result.dropna().values

            assert (values_left == values_right).all()
