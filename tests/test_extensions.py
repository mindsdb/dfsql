import modin.pandas as mpd
import pandas as pd
import pytest
from pdsql.exceptions import QueryExecutionException, pdsqlException


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(pd, id="pandas"),
        pytest.param(mpd, id="modin"),
    ],
)
class TestExtensions:
    def test_df_sql_simple_select(self, config, engine, csv_file):
        import pdsql.extensions

        df = engine.read_csv(csv_file)


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

    def test_df_sql_nested_select_in(self, config, engine, csv_file):
        import pdsql.extensions

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

    def test_df_sql_nested_select_from(self, config, engine, csv_file):
        import pdsql.extensions

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
