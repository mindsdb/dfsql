from dataskillet.sql_parser.select import Select
from dataskillet.sql_parser.constant import Constant
from dataskillet.sql_parser.expression import Expression, Star
from dataskillet.sql_parser.identifier import Identifier
from dataskillet.sql_parser.operation import BinaryOperation

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
                           args=(left, right),
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
    return Identifier(value=stmt['relname'], raw=stmt)


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


def parse_select_statement(select_stmt):
    print(select_stmt)

    targets = []
    for target in select_stmt['targetList']:
        targets.append(parse_statement(target['ResTarget']['val']))

    from_table = parse_statement(select_stmt.get('fromClause')[0]) if select_stmt.get('fromClause') else None

    where = parse_statement(select_stmt.get('whereClause')) if select_stmt.get('whereClause') else None

    group_by = None
    order_by = None
    limit = None
    return Select(raw=select_stmt,
                  targets=targets,
                  from_table=from_table,
                  where=where,
                  group_by=group_by,
                  order_by=order_by,
                  limit=limit)


def parse_sql(sql_query):
    sql_tree = pglast.parse_sql(sql_query)[0]
    try:
        select_statement = sql_tree['RawStmt']['stmt']['SelectStmt']
    except KeyError:
        raise Exception('SELECT excepted, but not found')

    out_tree = parse_select_statement(select_statement)
    return out_tree
