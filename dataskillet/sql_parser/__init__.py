from dataskillet.sql_parser.select import Select
from dataskillet.sql_parser.constant import Constant
from dataskillet.sql_parser.expression import Expression, Star
from dataskillet.sql_parser.identifier import Identifier
from dataskillet.sql_parser.operation import BinaryOperation, FunctionCall
from dataskillet.sql_parser.order_by import OrderBy, LOOKUP_ORDER_DIRECTIONS, LOOKUP_NULLS_SORT

import pglast


def parse_constant(stmt):
    dtype = next(iter(stmt['val'].keys()))

    value = None
    if dtype == 'Integer':
        value = int(stmt['val']['Integer']['ival'])
    elif dtype == 'Float':
        value = float(stmt['val']['Float']['str'])
    return Constant(value=value, raw=stmt)


def parse_expression(stmt):
    op = stmt['name'][0]['String']['str']
    left_stmt = stmt['lexpr']
    right_stmt = stmt['rexpr']

    left = parse_statement(left_stmt)
    right = parse_statement(right_stmt)
    return BinaryOperation(op=op,
                           args_=(left, right),
                           raw=stmt)


def parse_column_ref(stmt):
    field = next(iter(stmt['fields']))
    field_type = next(iter(field.keys()))
    if field_type == 'A_Star':
        return Star(raw=stmt)
    else:
        field = field[field_type]['str']
        return Identifier(value=field, raw=stmt)


def parse_rangevar(stmt):
    alias = None
    if stmt.get('alias'):
        alias = stmt['alias']['Alias']['aliasname']
    return Identifier(value=stmt['relname'], alias=alias, raw=stmt)


def parse_func_call(stmt):
    op = stmt['funcname'][0]['String']['str']
    args = [parse_statement(arg) for arg in stmt['args']]
    return FunctionCall(op=op,
                        args_=args,
                        raw=stmt)


def parse_statement(stmt):
    target_type = next(iter(stmt.keys()))
    if target_type == 'A_Const':
        return parse_constant(stmt['A_Const'])
    elif target_type == 'A_Expr':
        return parse_expression(stmt['A_Expr'])
    elif target_type == 'ColumnRef':
        return parse_column_ref(stmt['ColumnRef'])
    elif target_type == 'RangeVar':
        return parse_rangevar(stmt['RangeVar'])
    elif target_type == 'FuncCall':
        return parse_func_call(stmt['FuncCall'])


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


def parse_select_statement(select_stmt):
    print(select_stmt)

    targets = []
    for target in select_stmt['targetList']:
        targets.append(parse_target(target['ResTarget']))

    from_table = None
    if select_stmt.get('fromClause'):
        from_table = []
        for stmt in select_stmt['fromClause']:
            from_table.append(parse_statement(stmt))

    where = parse_statement(select_stmt.get('whereClause')) if select_stmt.get('whereClause') else None

    group_by = None
    if select_stmt.get('groupClause'):
        group_by = []
        for stmt in select_stmt['groupClause']:
            group_by.append(parse_statement(stmt))

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
                  from_table=from_table,
                  where=where,
                  group_by=group_by,
                  order_by=order_by,
                  limit=limit,
                  offset=offset)


def parse_sql(sql_query):
    sql_tree = pglast.parse_sql(sql_query)[0]
    try:
        select_statement = sql_tree['RawStmt']['stmt']['SelectStmt']
    except KeyError:
        raise Exception('SELECT excepted, but not found')

    out_tree = parse_select_statement(select_statement)
    return out_tree
