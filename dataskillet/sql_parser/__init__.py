from dataskillet.functions import AGGREGATE_MAPPING
from dataskillet.sql_parser.list_ import List
from dataskillet.sql_parser.select import Select
from dataskillet.sql_parser.constant import Constant
from dataskillet.sql_parser.expression import Expression, Star
from dataskillet.sql_parser.identifier import Identifier
from dataskillet.sql_parser.operation import Operation, BinaryOperation, Function, AggregateFunction, LOOKUP_BOOL_OPERATION, \
    InOperation, UnaryOperation, operation_factory, LOOKUP_NULL_TEST, ComparisonPredicate, LOOKUP_BOOL_TEST
from dataskillet.sql_parser.order_by import OrderBy, LOOKUP_ORDER_DIRECTIONS, LOOKUP_NULLS_SORT
from dataskillet.sql_parser.join import Join, LOOKUP_JOIN_TYPE
from dataskillet.exceptions import SQLParsingException
from dataskillet.commands import command_types


import pglast

from dataskillet.sql_parser.type_cast import TypeCast, MAP_DTYPES


def parse_constant(stmt):
    dtype = next(iter(stmt['val'].keys()))

    value = None
    if dtype == 'Integer':
        value = int(stmt['val']['Integer']['ival'])
    elif dtype == 'Float':
        value = float(stmt['val']['Float']['str'])
    elif dtype == "String":
        value = str(stmt['val']['String']['str'])
    return Constant(value=value, raw=stmt)


def parse_expression(stmt):
    op = stmt['name'][0]['String']['str']
    if stmt['kind'] == 7:
        op = 'IN'
    args = []
    if stmt.get('lexpr'):
        left_stmt = stmt['lexpr']
        left = parse_statement(left_stmt)
        args.append(left)

    if stmt.get('rexpr'):
        right_stmt = stmt['rexpr']
        right = parse_statement(right_stmt)
        args.append(right)

    return operation_factory(op=op, args=args, raw=stmt)


def parse_column_ref(stmt):
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


def parse_rangevar(stmt):
    alias = None
    if stmt.get('alias'):
        alias = stmt['alias']['Alias']['aliasname']
    return Identifier(value=stmt['relname'], alias=alias, raw=stmt)


def parse_func_call(stmt):
    op = stmt['funcname'][0]['String']['str']
    if stmt.get('agg_distinct'):
        op += '_distinct'
    class_ = Function
    if op in AGGREGATE_MAPPING:
        class_ = AggregateFunction
    args = [parse_statement(arg) for arg in stmt['args']]
    return class_(op=op,
                        args_=args,
                        raw=stmt)


def parse_bool_expr(stmt):
    op = LOOKUP_BOOL_OPERATION[stmt['boolop']]
    args = [parse_statement(arg) for arg in stmt['args']]

    def nested_ops(op, args):
        if len(args) > 2:
            return operation_factory(op, args=(args[0], nested_ops(op, args[1:])))
        return operation_factory(op, args=args)

    return nested_ops(op, args)


def parse_sublink(stmt):
    sublink_type = stmt['subLinkType']
    subselect = stmt['subselect']
    subselect = parse_select_statement(subselect)

    if sublink_type == 2:
        # IN clause
        leftarg = parse_statement(stmt['testexpr'])
        return InOperation(
            args_ = (
                leftarg,
                subselect
            ),
            raw=stmt,
        )
    else:
        return subselect


def parse_booltest(stmt):
    arg = parse_statement(stmt['arg'])
    op = LOOKUP_BOOL_TEST[stmt['booltesttype']]
    return ComparisonPredicate(op=op,
                          args_=(arg,),
                          raw=stmt)


def parse_nulltest(stmt):
    arg = parse_statement(stmt['arg'])
    op = LOOKUP_NULL_TEST[stmt['nulltesttype']]
    return ComparisonPredicate(op=op,
                          args_=(arg,),
                          raw=stmt)


def parse_typecast(stmt):
    arg = parse_statement(stmt['arg'])
    type_name = [name['String']['str'] for name in stmt['typeName']['TypeName']['names']][-1].lower()
    if MAP_DTYPES.get(type_name):
        type_name = MAP_DTYPES.get(type_name)
    return TypeCast(type_name=type_name, arg=arg, raw=stmt)


def parse_list(stmt):
    return List(tuple([parse_statement(item) for item in stmt]), raw=stmt)


def parse_statement(stmt):
    if isinstance(stmt, list):
        return parse_list(stmt)

    if isinstance(stmt, dict):
        target_type = next(iter(stmt.keys()))
        if target_type == 'A_Const':
            return parse_constant(stmt['A_Const'])
        elif target_type == 'A_Expr':
            return parse_expression(stmt['A_Expr'])
        elif target_type == 'BoolExpr':
            return parse_bool_expr(stmt['BoolExpr'])
        elif target_type == 'ColumnRef':
            return parse_column_ref(stmt['ColumnRef'])
        elif target_type == 'RangeVar':
            return parse_rangevar(stmt['RangeVar'])
        elif target_type == 'FuncCall':
            return parse_func_call(stmt['FuncCall'])
        elif target_type == 'SubLink':
            return parse_sublink(stmt['SubLink'])
        elif target_type == 'NullTest':
            return parse_nulltest(stmt['NullTest'])
        elif target_type == 'BooleanTest':
            return parse_booltest(stmt['BooleanTest'])
        elif target_type == 'TypeCast':
            return parse_typecast(stmt['TypeCast'])

    raise SQLParsingException(f'No idea how to parse {str(stmt)}')


def parse_order_by(stmt):
    field = parse_statement(stmt['node'])
    sort_dir = LOOKUP_ORDER_DIRECTIONS[stmt['sortby_dir']]
    sort_nulls = LOOKUP_NULLS_SORT[stmt['sortby_nulls']]
    return OrderBy(field=field,
                   direction=sort_dir,
                   nulls=sort_nulls,
                   raw=stmt)


def parse_target(stmt):
    val = stmt['val']
    alias = stmt.get('name')
    result = parse_statement(val)
    result.alias = alias
    return result


def parse_join(stmt):
    join_expr = stmt['JoinExpr']
    join_type = LOOKUP_JOIN_TYPE[join_expr['jointype']]
    left = parse_statement(join_expr['larg'])
    right = parse_statement(join_expr['rarg'])
    condition = parse_statement(join_expr['quals'])
    return Join(join_type=join_type,
                left=left,
                right=right,
                condition=condition,
                raw=stmt)


def parse_from_clause(stmt):
    from_table = []
    for from_clause in stmt:
        target_type = next(iter(from_clause.keys()))
        if target_type == 'JoinExpr':
            from_table.append(parse_join(from_clause))
        elif target_type == 'RangeSubselect':
            alias = from_clause['RangeSubselect']['alias']['Alias']['aliasname']
            subquery = parse_select_statement(from_clause['RangeSubselect']['subquery'])
            subquery.alias = alias
            from_table.append(subquery)
        else:
            from_table.append(parse_statement(from_clause))
    return from_table


def parse_select_statement(select_stmt):
    # print(select_stmt)
    select_stmt = select_stmt['SelectStmt']

    targets = []
    for target in select_stmt['targetList']:
        targets.append(parse_target(target['ResTarget']))

    distinct = select_stmt.get('distinctClause', None) is not None

    from_table = None
    if select_stmt.get('fromClause'):
        from_table = parse_from_clause(select_stmt['fromClause'])

    where = parse_statement(select_stmt.get('whereClause')) if select_stmt.get('whereClause') else None

    group_by = None
    if select_stmt.get('groupClause'):
        group_by = []
        for stmt in select_stmt['groupClause']:
            group_by.append(parse_statement(stmt))

    having = None
    if select_stmt.get('havingClause'):
        having = parse_statement(select_stmt['havingClause'])

    order_by = None
    if select_stmt.get('sortClause'):
        order_by = []
        for stmt in select_stmt['sortClause']:
            order_by.append(parse_order_by(stmt['SortBy']))

    offset = None
    if select_stmt.get('limitOffset'):
        offset = parse_statement(select_stmt['limitOffset'])

    limit = None
    if select_stmt.get('limitCount'):
        limit = parse_statement(select_stmt['limitCount'])

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


def parse_sql(sql_query):
    sql_tree = pglast.parse_sql(sql_query)
    if len(sql_tree) != 1:
        raise SQLParsingException('One SELECT statment expected')
    sql_tree = sql_tree[0]
    try:
        select_statement = sql_tree['RawStmt']['stmt']
    except KeyError:
        raise SQLParsingException('SELECT excepted, but not found')

    out_tree = parse_select_statement(select_statement)
    return out_tree


def try_parse_command(sql_query):
    for command_type in command_types:
        command = command_type.from_string(sql_query)

        if command:
            return command
