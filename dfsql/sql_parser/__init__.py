from dfsql.functions import AGGREGATE_MAPPING, is_supported
from dfsql.sql_parser.list_ import List
from dfsql.sql_parser.select import Select
from dfsql.sql_parser.constant import Constant
from dfsql.sql_parser.expression import Expression, Star
from dfsql.sql_parser.identifier import Identifier
from dfsql.sql_parser.operation import Operation, BinaryOperation, Function, AggregateFunction, LOOKUP_BOOL_OPERATION, \
    InOperation, UnaryOperation, operation_factory, LOOKUP_NULL_TEST, ComparisonPredicate, LOOKUP_BOOL_TEST
from dfsql.sql_parser.order_by import OrderBy, LOOKUP_ORDER_DIRECTIONS, LOOKUP_NULLS_SORT
from dfsql.sql_parser.join import Join, LOOKUP_JOIN_TYPE
from dfsql.exceptions import SQLParsingException
from dfsql.commands import command_types


import pglast

from dfsql.sql_parser.type_cast import TypeCast, MAP_DTYPES


class SQLParser:
    def __init__(self, custom_functions=None):
        self.custom_functions = custom_functions or {}

    def parse_constant(self, stmt):
        dtype = next(iter(stmt['val'].keys()))

        value = None
        if dtype == 'Integer':
            value = int(stmt['val']['Integer']['ival'])
        elif dtype == 'Float':
            value = float(stmt['val']['Float']['str'])
        elif dtype == "String":
            value = str(stmt['val']['String']['str'])
        return Constant(value=value, raw=stmt)

    def parse_expression(self, stmt):
        op = stmt['name'][0]['String']['str']
        if stmt['kind'] == 7:
            op = 'IN'
        args = []
        if stmt.get('lexpr'):
            left_stmt = stmt['lexpr']
            left = self.parse_statement(left_stmt)
            args.append(left)

        if stmt.get('rexpr'):
            right_stmt = stmt['rexpr']
            right = self.parse_statement(right_stmt)
            args.append(right)

        return operation_factory(op=op, args=args, raw=stmt)

    def parse_column_ref(self, stmt):
        fields = stmt['fields']
        field_type = next(iter(fields[0].keys()))
        if field_type == 'A_Star':
            return Star(raw=stmt)
        else:
            value = []
            for field in fields:
                field_type = next(iter(field.keys()))
                value.append(field[field_type]['str'])
            value = '.'.join(value)
            return Identifier(value=value, raw=stmt)

    def parse_rangevar(self, stmt):
        alias = None
        if stmt.get('alias'):
            alias = stmt['alias']['Alias']['aliasname']
        schema = stmt.get('schemaname', '')
        name = ((schema + '.') if schema else '') + stmt['relname']
        return Identifier(value=name, alias=alias, raw=stmt)

    def parse_func_call(self, stmt):
        op = stmt['funcname'][0]['String']['str']
        if stmt.get('agg_distinct'):
            op += '_distinct'
        class_ = Function

        if not (op in self.custom_functions) and not is_supported(op):
            raise SQLParsingException(f'Unknown operation {op}')

        if op in AGGREGATE_MAPPING:
            class_ = AggregateFunction
        args = [self.parse_statement(arg) for arg in stmt['args']]
        return class_(op=op,
                        args_=args,
                        raw=stmt)

    def parse_bool_expr(self, stmt):
        op = LOOKUP_BOOL_OPERATION[stmt['boolop']]
        args = [self.parse_statement(arg) for arg in stmt['args']]

        def nested_ops(op, args):
            if len(args) > 2:
                return operation_factory(op, args=(args[0], nested_ops(op, args[1:])))
            return operation_factory(op, args=args)

        return nested_ops(op, args)


    def parse_sublink(self, stmt):
        sublink_type = stmt['subLinkType']
        subselect = stmt['subselect']
        subselect = self.parse_select_statement(subselect)

        if sublink_type == 2:
            # IN clause
            leftarg = self.parse_statement(stmt['testexpr'])
            return InOperation(
                args_ = (
                    leftarg,
                    subselect
                ),
                raw=stmt,
            )
        else:
            return subselect

    def parse_booltest(self, stmt):
        arg = self.parse_statement(stmt['arg'])
        op = LOOKUP_BOOL_TEST[stmt['booltesttype']]
        return ComparisonPredicate(op=op,
                              args_=(arg,),
                              raw=stmt)

    def parse_nulltest(self, stmt):
        arg = self.parse_statement(stmt['arg'])
        op = LOOKUP_NULL_TEST[stmt['nulltesttype']]
        return ComparisonPredicate(op=op,
                              args_=(arg,),
                              raw=stmt)

    def parse_typecast(self, stmt):
        arg = self.parse_statement(stmt['arg'])
        type_name = [name['String']['str'] for name in stmt['typeName']['TypeName']['names']][-1].lower()
        if MAP_DTYPES.get(type_name):
            type_name = MAP_DTYPES.get(type_name)
        return TypeCast(type_name=type_name, arg=arg, raw=stmt)

    def parse_list(self, stmt):
        return List(tuple([self.parse_statement(item) for item in stmt]), raw=stmt)

    def parse_statement(self, stmt):
        if isinstance(stmt, list):
            return self.parse_list(stmt)

        if isinstance(stmt, dict):
            target_type = next(iter(stmt.keys()))
            if target_type == 'A_Const':
                return self.parse_constant(stmt['A_Const'])
            elif target_type == 'A_Expr':
                return self.parse_expression(stmt['A_Expr'])
            elif target_type == 'BoolExpr':
                return self.parse_bool_expr(stmt['BoolExpr'])
            elif target_type == 'ColumnRef':
                return self.parse_column_ref(stmt['ColumnRef'])
            elif target_type == 'RangeVar':
                return self.parse_rangevar(stmt['RangeVar'])
            elif target_type == 'FuncCall':
                return self.parse_func_call(stmt['FuncCall'])
            elif target_type == 'SubLink':
                return self.parse_sublink(stmt['SubLink'])
            elif target_type == 'NullTest':
                return self.parse_nulltest(stmt['NullTest'])
            elif target_type == 'BooleanTest':
                return self.parse_booltest(stmt['BooleanTest'])
            elif target_type == 'TypeCast':
                return self.parse_typecast(stmt['TypeCast'])

        raise SQLParsingException(f'No idea how to parse {str(stmt)}')

    def parse_order_by(self, stmt):
        field = self.parse_statement(stmt['node'])
        sort_dir = LOOKUP_ORDER_DIRECTIONS[stmt['sortby_dir']]
        sort_nulls = LOOKUP_NULLS_SORT[stmt['sortby_nulls']]
        return OrderBy(field=field,
                       direction=sort_dir,
                       nulls=sort_nulls,
                       raw=stmt)

    def parse_target(self, stmt):
        val = stmt['val']
        alias = stmt.get('name')
        result = self.parse_statement(val)
        result.alias = alias
        return result

    def parse_join(self, stmt):
        join_expr = stmt['JoinExpr']
        join_type = LOOKUP_JOIN_TYPE[join_expr['jointype']]
        left = self.parse_statement(join_expr['larg'])
        right = self.parse_statement(join_expr['rarg'])
        condition = self.parse_statement(join_expr['quals'])
        return Join(join_type=join_type,
                    left=left,
                    right=right,
                    condition=condition,
                    raw=stmt)

    def parse_from_clause(self, stmt):
        from_table = []
        for from_clause in stmt:
            target_type = next(iter(from_clause.keys()))
            if target_type == 'JoinExpr':
                from_table.append(self.parse_join(from_clause))
            elif target_type == 'RangeSubselect':
                alias = from_clause['RangeSubselect']['alias']['Alias']['aliasname']
                subquery = self.parse_select_statement(from_clause['RangeSubselect']['subquery'])
                subquery.alias = alias
                from_table.append(subquery)
            else:
                from_table.append(self.parse_statement(from_clause))
        return from_table

    def parse_select_statement(self, select_stmt):
        # print(select_stmt)
        select_stmt = select_stmt['SelectStmt']

        targets = []
        for target in select_stmt['targetList']:
            targets.append(self.parse_target(target['ResTarget']))

        distinct = select_stmt.get('distinctClause', None) is not None

        from_table = None
        if select_stmt.get('fromClause'):
            from_table = self.parse_from_clause(select_stmt['fromClause'])

        where = self.parse_statement(select_stmt.get('whereClause')) if select_stmt.get('whereClause') else None

        group_by = None
        if select_stmt.get('groupClause'):
            group_by = []
            for stmt in select_stmt['groupClause']:
                group_by.append(self.parse_statement(stmt))

        having = None
        if select_stmt.get('havingClause'):
            having = self.parse_statement(select_stmt['havingClause'])

        order_by = None
        if select_stmt.get('sortClause'):
            order_by = []
            for stmt in select_stmt['sortClause']:
                order_by.append(self.parse_order_by(stmt['SortBy']))

        offset = None
        if select_stmt.get('limitOffset'):
            offset = self.parse_statement(select_stmt['limitOffset'])

        limit = None
        if select_stmt.get('limitCount'):
            limit = self.parse_statement(select_stmt['limitCount'])

        return Select(raw=select_stmt,
                      targets=targets,
                      distinct=distinct,
                      from_table=from_table,
                      where=where,
                      group_by=group_by,
                      having=having,
                      order_by=order_by,
                      limit=limit,
                      offset=offset)


def parse_sql(sql_query, custom_functions=None):
    sql_tree = pglast.parse_sql(sql_query)
    if len(sql_tree) != 1:
        raise SQLParsingException('One SELECT statment expected')
    sql_tree = sql_tree[0]
    try:
        select_statement = sql_tree['RawStmt']['stmt']
    except KeyError:
        raise SQLParsingException('SELECT excepted, but not found')

    parser = SQLParser(custom_functions=custom_functions)
    out_tree = parser.parse_select_statement(select_statement)
    return out_tree


def try_parse_command(sql_query):
    for command_type in command_types:
        command = command_type.from_string(sql_query)

        if command:
            return command
