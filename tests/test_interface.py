import pandas as pd
import pytest

from dfsql.exceptions import QueryExecutionException, DfsqlException


class TestQuickInterface:
    def test_simple_select(self, csv_file):
        from dfsql.extensions import sql_query

        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM whatever_table AS new_table"

        query_result = sql_query(sql, whatever_table=df)
        assert query_result.name == 'passenger_id'
        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

        # Run query again to ensure that everything was cleaned up properly
        query_result = sql_query(sql, whatever_table=df)
        assert query_result.name == 'passenger_id'
        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

    def test_select_join(self, csv_file):
        from dfsql.extensions import sql_query
        df = pd.read_csv(csv_file)
        merge_df = pd.merge(df, df, how='inner', left_on=['passenger_id'], right_on=['p_class'])[
            ['passenger_id_x', 'p_class_y']]
        merge_df.columns = ['passenger_id', 'p_class']

        # Use one table for self join
        sql = "SELECT passenger_id, p_class FROM titanic AS t1 INNER JOIN titanic AS t2 ON t1.passenger_id = t2.p_class"
        query_result = sql_query(sql, titanic=df)

        assert list(query_result.columns) == ['passenger_id', 'p_class']
        values_left = merge_df[['passenger_id', 'p_class']].values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

        # Use two separate tables
        sql = "SELECT passenger_id, p_class FROM t1 INNER JOIN t2 ON t1.passenger_id = t2.p_class"
        query_result = sql_query(sql, t1=df, t2=df)

        assert list(query_result.columns) == ['passenger_id', 'p_class']
        values_left = merge_df[['passenger_id', 'p_class']].values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_error_table_not_found(self, csv_file):
        from dfsql.extensions import sql_query
        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM whatever_table INNER JOIN missing_table ON id"
        with pytest.raises(QueryExecutionException):
            sql_query(sql, whatever_table=df)

    def test_error_wrong_table_name(self, csv_file):
        from dfsql.extensions import sql_query

        df = pd.read_csv(csv_file)

        sql = "SELECT passenger_id FROM whatever_table"

        with pytest.raises(DfsqlException):
            sql_query(sql, wrong_table=df)

        # Run again to make sure it works after a failure
        query_result = sql_query(sql, whatever_table=df)
        assert query_result.name == 'passenger_id'
        values_left = df['passenger_id'].values
        values_right = query_result.values
        assert (values_left == values_right).all()

    def test_error_no_tables(self):
        from dfsql.extensions import sql_query
        sql = "SELECT passenger_id FROM whatever_table"

        with pytest.raises(DfsqlException):
            sql_query(sql, None)

        with pytest.raises(DfsqlException):
            sql_query(sql, something={})

        with pytest.raises(DfsqlException):
            sql_query(sql, something=[])

    def test_error_extra_tables(self, csv_file):
        from dfsql.extensions import sql_query
        df = pd.read_csv(csv_file)
        sql = "SELECT passenger_id FROM whatever_table"

        with pytest.raises(DfsqlException):
            sql_query(sql, whatever_table=df, extra_table=df)

    def test_custom_functions(self, csv_file):
        from dfsql.extensions import sql_query
        df = pd.read_csv(csv_file)
        sql = "SELECT sex, mode(survived) AS mode_survived FROM titanic GROUP BY sex"

        func = lambda x: x.value_counts(dropna=False).index[0]

        query_result = sql_query(sql, titanic=df, custom_functions={'mode': func})

        df = df.groupby(['sex']).agg({'survived': func}).reset_index()
        df.columns = ['sex', 'mode_survived']

        assert (query_result.columns == df.columns).all()
        assert query_result.shape == df.shape

        values_left = df.values
        values_right = query_result.values
        assert (values_left == values_right).all().all()

    def test_caps_column_names_dataframe(self, tmpdir):
        from dfsql.extensions import sql_query

        csv = """
ROUTE,DATE,RIDES
2,2021-02-27,3626
2,2021-02-28,5012
        """

        p = tmpdir.join('caps_df.csv')
        p.write_text(csv, encoding='utf-8')

        df = pd.read_csv(p)
        sql = """
SELECT `DATE` AS __timestamp,
       AVG(`RIDES`) AS `AVG(RIDES)`
FROM tab
GROUP BY `DATE`
ORDER BY `AVG(RIDES)` DESC
        """

        expected_output = df.groupby(['DATE']).agg({'RIDES': 'mean'}).reset_index()
        expected_output = expected_output.sort_values(by='RIDES', ascending=False)
        expected_output.columns = ['__timestamp', '`AVG(RIDES)`']

        query_result = sql_query(sql, tab=df)
        assert query_result.shape == expected_output.shape
        values_left = expected_output.dropna().values
        values_right = query_result.dropna().values

        assert (values_left == values_right).all()
