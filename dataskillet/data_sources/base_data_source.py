import modin.pandas as pd
import numpy as np
from dataskillet.sql_parser import (parse_sql, Select, Identifier, Constant, Operation, Star, Function,
                                    AggregateFunction, Join, BinaryOperation)


def get_modin_operation(sql_op):
    operations = {
        '+': sum,
        '-': lambda args: args[0] - args[1],
        '=': lambda args: args[0] == args[1],
        '!=': lambda args: args[0] != args[1],
        'avg': 'mean',
        'sum': 'sum',
        'count': 'count',
        'in': lambda args: args[0].isin(list(args[1])) if isinstance(args[0], pd.Series) or isinstance(args[0], pd.DataFrame) else args[0] in args[1]
    }
    op = operations.get(sql_op.lower())
    if not op:
        raise(Exception(f'Unsupported operation: {sql_op}'))
    return op


class DataSource:
    def __init__(self, tables):
        self.tables = {t.name.lower(): t for t in tables}

    def __contains__(self, table_name):
        return table_name in self.tables

    def add_table(self, table):
        if self.tables.get(table.name):
            raise Exception(f'Table {table.name} already exists in data source, use DROP TABLE to remove it if you want to recreate it.')
        self.tables[table.name] = table

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

    def execute_select_groupby_targets(self, targets, source_df, group_by):
        funcs_to_alias = {}

        out_column_names = []
        out_columns = []

        agg = {}

        group_by_cols = [q.value for q in group_by if q.value != True]
        for target in targets:
            col_df_name = target.alias
            col_name = target.alias
            if isinstance(target, Identifier):
                col_df_name = target.value
                col_name = target.alias if target.alias else target.value

            elif not target.alias:
                raise (Exception(f'Alias required for {target}'))

            if isinstance(target, Function):
                arg = target.args[0]
                assert isinstance(arg, Identifier)
                arg = arg.value
                modin_op = get_modin_operation(target.op)
                if agg.get(arg):
                    agg[arg].append(modin_op)
                else:
                    agg[arg] = [modin_op]

                funcs_to_alias[(arg, modin_op)] = col_name

            else:
                if col_name not in group_by_cols and col_df_name not in group_by_cols:
                    raise Exception(f'Column {col_df_name}({col_name}) not found in GROUP BY clause')

        aggregate_result = source_df.agg(agg)
        for col_index in aggregate_result.reset_index().columns:
            if isinstance(col_index, tuple):
                col_name = col_index[0]
            else:
                col_name = col_index
            if col_name in agg or col_name == 'index':
                continue

            out_column_names.append(col_name)
            out_columns.append(aggregate_result.reset_index()[col_name].values)

        for col_name in agg:
            for func in agg[col_name]:
                alias = funcs_to_alias[(col_name, func)]
                agg_result = aggregate_result[col_name][func]
                try:
                    agg_result = agg_result.values
                except AttributeError:
                    agg_result = np.array([agg_result])
                out_columns.append(agg_result)
                out_column_names.append(alias)

        out_dict = {col: values for col, values in zip(out_column_names, out_columns)}
        out_df = pd.DataFrame(out_dict)
        return out_df

    def execute_select(self, query):
        from_table = [self.execute_from_query(sub_q) for sub_q in query.from_table]

        if len(from_table) != 1:
            raise(Exception(f'No idea how to deal with from_table len {len(from_table)}'))

        source_df = from_table[0]
        if query.where:
            index = self.execute_operation(query.where, source_df)
            source_df = source_df[index]
        group_by = False

        if query.group_by is None:
            # Check for implicit group by
            non_agg_functions = []
            agg_functions = []
            for target in query.targets:
                if isinstance(target, AggregateFunction):
                    agg_functions.append(target)
                else:
                    non_agg_functions.append(target)

            if not non_agg_functions and agg_functions:
                query.group_by = [Constant(True)]
            elif non_agg_functions and agg_functions:
                raise(Exception(f'Can\'t process a mix of aggregation functions and non-aggregation functions with no GROUP BY clause.'))
        if query.group_by:
            group_by = True
            source_df = self.execute_groupby_queries(query.group_by, source_df)

        if group_by == False:
            out_df = self.execute_select_targets(query.targets, source_df)
        else:
            out_df = self.execute_select_groupby_targets(query.targets, source_df, query.group_by)

        if query.having:
            if group_by == False:
                raise Exception('Can\'t execute HAVING clause with no GROUP BY clause.')
            index = self.execute_operation(query.having, out_df)
            out_df = out_df[index]

        if query.distinct:
            out_df = out_df.drop_duplicates()

        if query.offset:
            offset = self.execute_query(query.offset)
            out_df = out_df.iloc[offset:, :]

        if query.limit:
            limit = self.execute_query(query.limit)
            out_df = out_df.iloc[:limit, :]

        if out_df.shape == (1, 1): # Just one value returned
            return out_df.values[0][0]
        elif out_df.shape[1] == 1: # Just one column, return series
            return out_df.iloc[:, 0]
        return out_df

    def execute_join(self, query):
        join_type = query.join_type
        join_type = {'INNER JOIN': 'inner', 'LEFT JOIN': 'left', 'RIGHT JOIN': 'right', 'FULL JOIN': 'outer'}[join_type]

        left = query.left
        if isinstance(left, Identifier):
            left = self.execute_from_identifier(left)
        else:
            left = self.execute_query(left)

        right = query.right
        if isinstance(right, Identifier):
            right = self.execute_from_identifier(right)
        else:
            right = self.execute_query(right)

        condition = query.condition
        if isinstance(condition, BinaryOperation):
            left_on = condition.args[0]
            right_on = condition.args[1]
        else:
            raise Exception(f'Invalid join condition {condition.op}')
        left_name = query.left.alias if query.left.alias else query.left.value
        right_name = query.right.alias if query.right.alias else query.right.value
        left_on, right_on = left_on if left_on.value.split('.')[0] in left_name else right_on, \
                            right_on if right_on.value.split('.')[0] in right_name else left_on

        left_on = left_on.value.split('.')[-1]
        right_on = right_on.value.split('.')[-1]
        out_df = pd.merge(left, right, how=join_type, left_on=[left_on], right_on=[right_on], suffixes=('_x', '_y'))
        out_df = out_df.drop(columns=[f'{left_on}_y', f'{right_on}_x'])

        renaming = {f'{left_on}_x': left_on, f'{right_on}_y': right_on}

        for col in out_df.columns:
            if col in renaming:
                continue

            if '_x' in col:
                pure_col_name = col.replace('_x', '')
                renaming[col] = f'{left_name}.{pure_col_name}'
            elif '_y' in col:
                pure_col_name = col.replace('_y', '')
                renaming[col] = f'{right_name}.{pure_col_name}'

        out_df = out_df.rename(renaming, axis=1)
        return out_df

    def execute_from_query(self, query):
        if isinstance(query, Identifier):
            return self.execute_from_identifier(query)
        if isinstance(query, Join):
            return self.execute_join(query)
        return self.execute_query(query)

    def execute_groupby_queries(self, queries, df):
        col_names = []

        if len(queries) == 1 and isinstance(queries[0], Constant) and queries[0].value == True:
            return df

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
