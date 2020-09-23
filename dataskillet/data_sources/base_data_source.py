import modin.pandas as pd
from dataskillet.sql_parser import parse_sql
from dataskillet.sql_parser import (Select, Identifier)


class DataSource:
    def __init__(self, tables):
        self.tables = {t.name.lower(): t for t in tables}

    def __contains__(self, table_name):
        return table_name in self.tables

    def query(self, sql):
        query = parse_sql(sql)

        return self.execute_query(query)

    def execute_table_identifier(self, query):
        table_name = query.value
        if not table_name in self:
            raise(Exception(f'Unknown table {table_name}'))
        else:
            return self.tables[table_name].df

    def execute_select_target(self, query, df):
        if isinstance(query, Identifier):
            return df[query.value.lower()]

        return self.execute_query(query)

    def execute_select(self, query):
        from_table = [self.execute_query(sub_q) for sub_q in query.from_table]

        if len(from_table) != 1:
            raise(Exception(f'No idea how to deal with from_table len {len(from_table)}'))

        source_df = from_table[0]

        out_column_names = []
        out_columns = []
        for target in query.targets:
            if isinstance(target, Identifier):
                out_column_names.append(target.alias if target.alias else target.value)
            else:
                if not target.alias:
                    raise(Exception(f'Alias required for {target}'))
                out_column_names.append(target.alias)
            out_columns.append(self.execute_select_target(target, source_df))

        out_dict = {col: values for col, values in zip(out_column_names, out_columns)}
        out_df = pd.DataFrame(out_dict)

        if query.distinct:
            out_df = out_df.drop_duplicates()

        if out_df.shape == (1, 1): # Just one value returned
            return out_df.values[0][0]
        elif out_df.shape[1] == 1: # Just one column, return series
            return out_df[out_df.columns[0]]

        return out_df

    def execute_query(self, query):
        if isinstance(query, Select):
            return self.execute_select(query)

        elif isinstance(query, Identifier):
            return self.execute_table_identifier(query)
        else:
            raise (Exception(f'Unexpected query thing {type(query)}'))
