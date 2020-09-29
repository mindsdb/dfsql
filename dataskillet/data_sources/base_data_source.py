import modin.pandas as pd
from dataskillet.sql_parser import parse_sql
from dataskillet.sql_parser import (Select, Identifier, Constant, Operation, Star, FunctionCall)


def get_modin_operation(sql_op):
    operations = {
        '+': sum,
        '-': lambda args: args[0] - args[1],
        '=': lambda args: args[0] == args[1],
        '!=': lambda args: args[0] != args[1],
    }

    op = operations.get(sql_op)
    if not op:
        raise(Exception(f'Unsupported operation: {sql_op}'))
    return op


class DataSource:
    def __init__(self, tables):
        self.tables = {t.name.lower(): t for t in tables}

    def __contains__(self, table_name):
        return table_name in self.tables

    def query(self, sql):
        query = parse_sql(sql)

        return self.execute_query(query)

    def execute_from_identifier(self, query):
        table_name = query.value
        if table_name not in self:
            raise(Exception(f'Unknown table {table_name}'))
        else:
            return self.tables[table_name].df

    def execute_groupby_identifier(self, query, df):
        col_name = query.value
        if col_name not in df.columns:
            raise(Exception(f'Column {col_name} not found in dataframe'))

        return col_name

    def execute_constant(self, query):
        value = query.value
        return value

    def execute_operation(self, query, df):
        args = [self.execute_select_target(arg, df) for arg in query.args]
        op_func = get_modin_operation(query.op)
        return op_func(args)

    def execute_select_target(self, query, df):
        if isinstance(query, Identifier):
            return df[query.value.lower()]
        elif isinstance(query, Operation):
            return self.execute_operation(query, df)

        return self.execute_query(query)

    def execute_select_targets(self, targets, source_df):
        out_column_names = []
        out_columns = []
        # Expand star
        for i, target in enumerate(targets):
            if isinstance(target, Star):
                targets = targets[:i] + [Identifier(colname) for colname in
                                                     source_df.columns] + targets[i + 1:]
                break

        for target in targets:
            if isinstance(target, Identifier):
                out_column_names.append(target.alias if target.alias else target.value)
            else:
                if not target.alias:
                    raise (Exception(f'Alias required for {target}'))
                out_column_names.append(target.alias)
            out_columns.append(self.execute_select_target(target, source_df))

        out_dict = {col: values for col, values in zip(out_column_names, out_columns)}
        out_df = pd.DataFrame(out_dict)
        return out_df

    def execute_select_groupby_targets(self, targets, source_df):
        agg = {}
        for target in targets:
            if isinstance(target, FunctionCall):
                arg = target.args[0]
                assert isinstance(arg, Identifier)
                arg = arg.value
                agg[arg] = target.op
        return source_df.agg(agg).reset_index()

    def execute_select(self, query):
        from_table = [self.execute_from_query(sub_q) for sub_q in query.from_table]

        if len(from_table) != 1:
            raise(Exception(f'No idea how to deal with from_table len {len(from_table)}'))

        source_df = from_table[0]

        if query.where:
            index = self.execute_operation(query.where, source_df)
            source_df = source_df[index]

        group_by = False
        if query.group_by:
            group_by = True
            source_df = self.execute_groupby_queries(query.group_by, source_df)

        if group_by == False:
            out_df = self.execute_select_targets(query.targets, source_df)
        else:
            out_df = self.execute_select_groupby_targets(query.targets, source_df)

        if query.distinct:
            out_df = out_df.drop_duplicates()

        if out_df.shape == (1, 1): # Just one value returned
            return out_df.values[0][0]
        elif out_df.shape[1] == 1: # Just one column, return series
            return out_df[out_df.columns[0]]

        return out_df

    def execute_from_query(self, query):
        if isinstance(query, Identifier):
            return self.execute_from_identifier(query)
        return self.execute_query(query)

    def execute_groupby_queries(self, queries, df):
        col_names = []
        for query in queries:
            if isinstance(query, Identifier):
                col_names.append(self.execute_groupby_identifier(query, df))
            else:
                raise Exception(f"Don't know how to aggregate by {str(query)}")
        return df.groupby(col_names)

    def execute_query(self, query):
        if isinstance(query, Select):
            return self.execute_select(query)
        elif isinstance(query, Constant):
            return self.execute_constant(query)
        else:
            raise (Exception(f'Unexpected query thing {type(query)}'))
