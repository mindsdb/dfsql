import os
from dfsql.engine import pd
import json

from dfsql.cache import MemoryCache
from dfsql.exceptions import QueryExecutionException
from dfsql.functions import OPERATION_MAPPING, AGGREGATE_MAPPING
from dfsql.commands import try_parse_command
from mindsdb_sql import parse_sql
from mindsdb_sql.parser.ast import (Select, Identifier, Constant, Operation, Function, Join, BinaryOperation, TypeCast,
                                    Tuple, NullConstant, Star)
from dfsql.table import Table, FileTable
from dfsql.utils import CaseInsensitiveDict, pd_get_column_case_insensitive, get_df_column, CaseInsensitiveKey


def get_modin_operation(sql_op):
    op = OPERATION_MAPPING.get(sql_op.lower())
    if not op:
        raise(QueryExecutionException(f'Unsupported operation: {sql_op}'))
    return op()


def get_aggregation_operation(sql_op):
    op = AGGREGATE_MAPPING.get(sql_op.lower())
    if not op:
        raise(QueryExecutionException(f'Unsupported operation: {sql_op}'))
    return op.string_or_callable()


def cast_type(obj, type_name):
    if not hasattr(obj, 'astype'):
        obj = pd.Series(obj)
    return obj.astype(type_name)


class DataSource:
    def __init__(self,
                 metadata_dir,
                 tables=None,
                 cache=None,
                 custom_functions=None,
                 case_sensitive=True):
        self.metadata_dir = metadata_dir

        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir, exist_ok=True)

        self.case_sensitive = case_sensitive


        tables = {t.name: t for t in tables} if tables else {}

        if not self.case_sensitive:
            tables = CaseInsensitiveDict(tables)

        self.tables = None
        self.load_metadata()
        if self.tables and not self.case_sensitive:
            self.tables = CaseInsensitiveDict(self.tables)

        if self.tables and tables:
            raise QueryExecutionException(f'Table metadata already exists in directory {metadata_dir}, but tables also passed to DataSource constructor. '
                            f'\nEither load the previous metadata by omitting the tables argument, or explicitly overwrite old metadata by using DataSource.create_new(metadata_dir, tables).')
        elif not self.tables:
            self.tables = tables

        self.save_metadata()

        self.set_cache(cache or MemoryCache())

        self._query_scope = set()
        
        self.custom_functions = custom_functions or {}


    def set_cache(self, cache):
        self.cache = cache
        for tname, table in self.tables.items():
            table.cache = self.cache

    @property
    def query_scope(self):
        """Stores aliases and tables available to a select during execution"""
        return self._query_scope

    def clear_query_scope(self):
        self._query_scope = set()

    @classmethod
    def create_new(cls, metadata_dir, tables=None):
        cls.clear_metadata(metadata_dir)
        return cls(metadata_dir, tables=tables)

    @classmethod
    def clear_metadata(cls, metadata_dir):
        if os.path.exists(os.path.join(metadata_dir, 'datasource_tables.json')):
            os.remove(os.path.join(metadata_dir, 'datasource_tables.json'))

    def add_table_from_file(self, path):
        table = FileTable.from_file(path)
        self.add_table(table)

    @staticmethod
    def from_dir(metadata_dir, files_dir_path, *args, **kwargs):
        files = os.listdir(files_dir_path)
        ds = DataSource(*args, metadata_dir=metadata_dir, **kwargs)
        for f in files:
            if f.endswith('.csv'):
                fpath = os.path.join(files_dir_path, f)
                ds.add_table_from_file(fpath)

        if not ds.tables:
            raise(QueryExecutionException(f'Directory {files_dir_path} does not contain any spreadsheet files'))
        return ds

    def load_metadata(self):
        if not os.path.exists(os.path.join(self.metadata_dir, 'datasource_tables.json')):
            return

        new_tables = {}
        with open(os.path.join(self.metadata_dir, 'datasource_tables.json'), 'r') as f:
            table_data = json.load(f)

        for tname, table_json in table_data.items():
            new_tables[tname] = Table.from_json(table_json)

        self.tables = new_tables

    def save_metadata(self, overwrite=True):
        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)

        if not os.access(self.metadata_dir, os.W_OK):
            raise QueryExecutionException(f'Directory {self.metadata_dir} not writable')

        tables_dump = {
            tname: table.to_json() for tname, table in self.tables.items()
        }

        if not overwrite and os.path.exists(os.path.join(self.metadata_dir, 'datasource_tables.json')):
            raise QueryExecutionException('Table metadata already exists, but overwrite is False.')

        with open(os.path.join(self.metadata_dir, 'datasource_tables.json'), 'w') as f:
            f.write(json.dumps(tables_dump))

    def __contains__(self, table_name):
        return table_name in self.tables

    def register_function(self, name, func):
        self.custom_functions[name] = func

    def add_table(self, table):
        if self.tables.get(table.name):
            raise QueryExecutionException(f'Table {table.name} already exists in data source, use DROP TABLE to remove it if you want to recreate it.')
        self.tables[table.name] = table
        self.save_metadata()

    def drop_table(self, name):
        del self.tables[name]
        self.save_metadata()

    def execute_command(self, command):
        return command.execute(self)

    def query(self, sql, reduce_output=True):
        command = try_parse_command(sql)
        if command:
            return self.execute_command(command)
        query = parse_sql(sql)
        return self.execute_query(query, reduce_output=reduce_output)

    def execute_table_identifier(self, query):
        table_name = query.parts_to_str()
        if table_name not in self:
            raise QueryExecutionException(f'Unknown table {table_name}')
        else:
            df = self.tables[table_name].dataframe
            self.query_scope.add(table_name)
            return df

    def execute_constant(self, query):
        if isinstance(query, NullConstant):
            return None
        value = query.value
        return value

    def execute_operation(self, query, df):
        args = [self.execute_select_target(arg, df) for arg in query.args]
        op_func = self.custom_functions.get(query.op.lower())
        if not op_func:
            op_func = get_modin_operation(query.op.lower())
        result = op_func(*args)
        return result

    def execute_column_identifier(self, query, df):
        name_components = query.parts

        if len(name_components) == 1:
            full_column_name = name_components[0]
            column = get_df_column(df, full_column_name, case_sensitive=self.case_sensitive)
            if column is not None:
                return column
        elif len(name_components) == 2:
            table_name, column_name = name_components

            # If it's a join or a subquery
            join_column_name = f'{table_name}.{column_name}'
            column = get_df_column(df, join_column_name, case_sensitive=self.case_sensitive)
            if column is not None:
                return column

            if table_name and not table_name in self.query_scope:
                raise QueryExecutionException(f"Table name {table_name} not in scope.")

            column = get_df_column(df, column_name, case_sensitive=self.case_sensitive)
            if column is not None:
                column.name = query.parts_to_str()
                return column
        else:
            raise QueryExecutionException(f"Too many name components: {query.parts}")
        raise QueryExecutionException(f"Column {query.parts_to_str()} not found.")

    def execute_type_cast(self, query, df):
        type_name = query.type_name
        arg = self.execute_select_target(query.arg, df)
        return cast_type(arg, type_name)

    def execute_select_target(self, query, df):
        if isinstance(query, Identifier):
            return self.execute_column_identifier(query, df)
        elif isinstance(query, Operation):
            return self.execute_operation(query, df)
        elif isinstance(query, TypeCast):
            return self.execute_type_cast(query, df)

        return self.execute_query(query, reduce_output=True)

    def resolve_select_target_col_name(self, target):
        if not target.alias:
            col_name = target.to_string(alias=False)
        else:
            col_name = target.alias.to_string(alias=False)
        return col_name

    def execute_select_targets(self, targets, source_df):
        out_names = []

        iterable_names = []
        iterable_columns = []

        scalar_names = []
        scalar_values = []
        # Expand star
        for i, target in enumerate(targets):
            if isinstance(target, Star):
                targets = targets[:i] + [Identifier(colname) for colname in
                                                     source_df.columns] + targets[i + 1:]
                break

        for target in targets:
            col_name = self.resolve_select_target_col_name(target)
            out_names.append(col_name)

            select_target_result = self.execute_select_target(target, source_df)
            if isinstance(select_target_result, pd.Series):
                iterable_names.append(col_name)
                iterable_columns.append(select_target_result)
            else:
                scalar_names.append(col_name)
                scalar_values.append(select_target_result)

        # Add columns first, then scalars, so the dataframe has proper index in the end
        out_columns = {}
        for i, col_name in enumerate(iterable_names):
            out_columns[col_name] = list(iterable_columns[i])
        out_df = pd.DataFrame.from_dict(out_columns)
        for i, col_name in enumerate(scalar_names):
            if out_df.empty:
                out_df[col_name] = [scalar_values[i]]
            else:
                out_df[col_name] = scalar_values[i]
        out_df = out_df[out_names]
        return out_df

    def execute_select_groupby_targets(self, targets, source_df, group_by):
        target_column_names = [] # Original names of columns to be returned by group by
        agg = {} # Agg dict for pandas aggregation

        column_renames = {} # Aliases for columns to be returned

        df_columns = getattr(source_df, '_columns', None)
        if df_columns is None:
            df_columns = source_df.columns

        df_original_column_names_lookup = dict(zip(df_columns, df_columns))
        if not self.case_sensitive:
            column_renames = CaseInsensitiveDict(column_renames)
            df_original_column_names_lookup = CaseInsensitiveDict(df_original_column_names_lookup)

        # Obtain columns that aggregation happens by
        group_by_cols = [] # Columns that aggregation happens over. Only these can be among targets and not under an agg func
        for g in group_by:
            if isinstance(g, Identifier) or isinstance(g, Operation):
                string_repr = g.to_string(alias=False)
                if not self.case_sensitive:
                    string_repr = CaseInsensitiveKey(string_repr)
                group_by_cols.append(string_repr)
            elif isinstance(g, Operation):
                if self.case_sensitive:
                    group_by_cols.append(str(g))
                else:
                    group_by_cols.append(CaseInsensitiveKey(str(g)))
            elif g == Constant(True): # Special case of implicit aggregation
                continue
            else:
                raise QueryExecutionException(f'Dont know how to handle group by column: {str(g)}')

        # Obtain column names, column aliases and aggregations to perform
        for target in targets:
            col_name = target.to_string(alias=False)
            if not self.case_sensitive:
                col_name = CaseInsensitiveKey(col_name)

            if target.alias:
                column_renames[col_name] = target.alias.to_string(alias=False)

            target_column_names.append(col_name)

            if col_name in agg:
                raise QueryExecutionException(f'Duplicate column name {col_name}. Provide an alias to resolve ambiguity.')

            if isinstance(target, Function):
                if col_name in group_by_cols:
                    # It's not a function to be executed, it's a transformed column from group by clause, leave it be
                    continue

                if len(target.args) > 1:
                    raise QueryExecutionException(f'Only one argument functions supported for aggregations, found: {str(target)}')

                arg = target.args[0]
                if not isinstance(arg, Identifier):
                    raise QueryExecutionException(f'The argument of an aggregate function must be a column, found: {str(arg)}')
                arg_col = arg.parts_to_str()

                func_name = target.op.lower()
                if target.distinct:
                    func_name = f'{target.op.lower()}_distinct'
                modin_op = self.custom_functions.get(func_name)
                if not modin_op:
                    modin_op = get_aggregation_operation(func_name)

                arg_col_name = df_original_column_names_lookup[arg_col]
                agg[str(col_name)] = (arg_col_name, modin_op)
            elif col_name not in group_by_cols:
                # Not a function to be executed and not found among groupbys, sus
                raise QueryExecutionException(f'Column {col_name} not found in GROUP BY clause')

        if isinstance(source_df, pd.Series):
            source_df = pd.DataFrame(source_df)

        if isinstance(source_df, pd.DataFrame):
            # If it's an implicit aggregation
            temp_df = pd.DataFrame(source_df)
            temp_df['__dummy__'] = 0
            source_df = temp_df.groupby('__dummy__')

        # Perform aggregation
        aggregate_result = source_df.agg(**agg)

        out_df_column_names = []
        out_df_column_values = []
        for col_index in aggregate_result.reset_index().columns:
            if col_index not in target_column_names:
                continue
            column_name = column_renames.get(col_index, col_index)
            out_df_column_names.append(column_name)
            out_df_column_values.append(aggregate_result.reset_index()[col_index].values)

        out_dict = {col: values for col, values in zip(out_df_column_names, out_df_column_values)}
        out_df = pd.DataFrame.from_dict(out_dict)
        return out_df

    def execute_order_by(self, order_by, df):
        fields = [s.field.parts_to_str() for s in order_by]
        sort_orders = [s.direction != 'DESC' for s in order_by]
        df = df.sort_values(by=fields, ascending=sort_orders)
        return df

    def execute_select(self, query, reduce_output=False):
        from_table = []
        if query.from_table:
            from_table = self.execute_from_query(query.from_table)

        source_df = from_table

        if query.where:
            index = self.execute_operation(query.where, source_df)
            source_df = source_df[index.values]
        group_by = False

        if query.group_by is None:
            # Check for implicit group by
            non_agg_functions = []
            agg_functions = []
            for target in query.targets:
                if isinstance(target, Function) and target.op.lower() in AGGREGATE_MAPPING:
                    agg_functions.append(target)
                else:
                    non_agg_functions.append(target)

            if not non_agg_functions and agg_functions:
                query.group_by = [Constant(True)]
            elif non_agg_functions and agg_functions:
                raise(QueryExecutionException(f'Can\'t process a mix of aggregation functions and non-aggregation functions with no GROUP BY clause.'))
        if query.group_by:
            group_by = True
            source_df = self.execute_groupby_queries(query.group_by, source_df)

        if group_by == False:
            out_df = self.execute_select_targets(query.targets, source_df)
        else:
            out_df = self.execute_select_groupby_targets(query.targets, source_df, query.group_by)

        if query.having:
            if group_by == False:
                raise QueryExecutionException('Can\'t execute HAVING clause with no GROUP BY clause.')
            index = self.execute_operation(query.having, out_df)
            out_df = out_df[index]

        if query.distinct:
            out_df = out_df.drop_duplicates()

        if query.offset:
            offset = self.execute_query(query.offset)
            out_df = out_df.iloc[offset:, :]

        if query.order_by:
            out_df = self.execute_order_by(query.order_by, out_df)

        if query.limit:
            limit = self.execute_query(query.limit)
            out_df = out_df.iloc[:limit, :]

        self.clear_query_scope()

        #Postprocess column names
        new_cols = []
        for col in out_df.columns:
            if col.startswith('`') and col.endswith('`') and not '.' in col:
                new_cols.append(col.strip('`'))
            else:
                new_cols.append(col)
        out_df.columns = new_cols

        # Turn tables into Series or constants if needed, for final returning
        if reduce_output:
            if out_df.shape == (1, 1): # Just one value returned
                return out_df.values[0][0]
            elif out_df.shape[1] == 1:
                return out_df[out_df.columns[0]]
        return out_df

    def execute_join(self, query):
        join_type = query.join_type
        join_type = {'INNER JOIN': 'inner', 'LEFT JOIN': 'left', 'RIGHT JOIN': 'right', 'FULL JOIN': 'outer'}[join_type]

        left = query.left
        if isinstance(left, Identifier):
            left = self.execute_table_identifier(left)
        else:
            left = self.execute_query(left)

        right = query.right
        if isinstance(right, Identifier):
            right = self.execute_table_identifier(right)
        else:
            right = self.execute_query(right)

        condition = query.condition
        if isinstance(condition, BinaryOperation):
            left_on = condition.args[0]
            right_on = condition.args[1]
        else:
            raise QueryExecutionException(f'Invalid join condition {condition.op}')
        left_name = query.left.alias.to_string(alias=False) if query.left.alias else query.left.to_string(alias=False)
        right_name = query.right.alias.to_string(alias=False) if query.right.alias else query.right.to_string(alias=False)
        left_on, right_on = left_on if left_on.parts[0] in left_name else right_on, \
                            right_on if right_on.parts[0] in right_name else left_on

        left_on = left_on.parts[-1]
        right_on = right_on.parts[-1]
        out_df = pd.merge(left, right, how=join_type, left_on=[left_on], right_on=[right_on], suffixes=('_x', '_y'))
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
            df = self.execute_table_identifier(query)
        elif isinstance(query, Join):
            df = self.execute_join(query)
        else:
            df = self.execute_query(query)

        if query.alias:
            self.query_scope.add(query.alias.to_string(alias=False))

        return df

    def execute_groupby_queries(self, queries, df):
        col_names = []

        if len(queries) == 1 and queries[0] == Constant(True):
            return df

        for query in queries:
            if isinstance(query, Identifier):
                column = self.execute_column_identifier(query, df)
                col_names.append(column.name)
            elif isinstance(query, Operation):
                expr_result = self.execute_operation(query, df)
                temp_col_name = query.alias if hasattr(query, 'alias') and query.alias else str(query)
                df[temp_col_name] = expr_result
                col_names.append(temp_col_name)
            else:
                raise QueryExecutionException(f"Don't know how to aggregate by {str(query)}")
        return df.groupby(col_names)

    def execute_query(self, query, reduce_output=False):
        if isinstance(query, Select):
            return self.execute_select(query, reduce_output=reduce_output)
        elif isinstance(query, Constant):
            return self.execute_constant(query)
        elif isinstance(query, Tuple):
            return pd.Series([self.execute_query(item) for item in query.items])
        else:
            raise QueryExecutionException(f'No idea how to execute query statement {type(query)}')
