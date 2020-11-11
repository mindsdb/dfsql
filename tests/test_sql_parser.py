import pytest
from dataskillet.sql_parser import (parse_sql, Select, Constant, Star, Identifier, BinaryOperation, Function,
                                    OrderBy, Join, InOperation, SQLParsingException, UnaryOperation, ComparisonPredicate)


class TestParseSelect:

    def test_no_select(self):
        query = ""
        with pytest.raises(SQLParsingException):
            parse_sql(query)

    def test_basic_select(self):
        query = """SELECT 1, 2.0"""

        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Constant(1), Constant(2.0)]))

    def test_select_from(self):
        query = """SELECT *, column1, column1 as aliased, column1 + column2 FROM t1"""

        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star(),
                                                            Identifier("column1"),
                                                            Identifier("column1", alias='aliased'),
                                                            BinaryOperation(op="+",
                                                                            args_=(Identifier('column1'),
                                                                                   Identifier('column2'))
                                                                            )
                                                            ],
                                                   from_table=[Identifier('t1')]))

    def test_select_distinct(self):
        query = """SELECT DISTINCT column1 FROM t1"""
        assert str(parse_sql(query)) == query

    def test_select_from_aliased(self):
        query = """SELECT * FROM t1 as t2"""
        assert str(parse_sql(query)) == query

    def test_select_from_where(self):
        query = """SELECT column1, column2 FROM t1 WHERE column1 = 1"""

        assert str(parse_sql(query)) == query

        assert str(parse_sql(query)) == str(Select(targets=[Identifier("column1"), Identifier("column2")],
                                                   from_table=[Identifier('t1')],
                                                   where=BinaryOperation(op="=",
                                                                         args_=(Identifier('column1'), Constant(1))
                                                                         )))

    def test_select_group_by(self):
        query = """SELECT column1, column2, sum(column3) as total FROM t1 GROUP BY column1, column2"""

        assert str(parse_sql(query)) == query

        assert str(parse_sql(query)) == str(Select(targets=[Identifier("column1"),
                                                            Identifier("column2"),
                                                            Function(op="sum",
                                                                         args_=(Identifier("column3"),),
                                                                         alias='total')],
                                                   from_table=[Identifier('t1')],
                                                   group_by=[Identifier("column1"), Identifier("column2")]))

    def test_select_groupby_having(self):
        query = """SELECT column1 FROM t1 GROUP BY column1 HAVING column1 <> 1"""
        assert str(parse_sql(query)) == query

    def test_select_order_by(self):
        query = """SELECT * FROM t1 ORDER BY column1 ASC, column2, column3 DESC NULLS FIRST"""
        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star()],
                                                   from_table=[Identifier('t1')],
                                                   order_by=[
                                                       OrderBy(Identifier('column1'), direction='ASC'),
                                                       OrderBy(Identifier('column2')),
                                                       OrderBy(Identifier('column3'), direction='DESC',
                                                               nulls='NULLS FIRST')],
                                                   ))

    def test_select_limit_offset(self):
        query = """SELECT * FROM t1 LIMIT 1 OFFSET 2"""
        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star()],
                                                   from_table=[Identifier('t1')],
                                                   limit=Constant(1),
                                                   offset=Constant(2)))

    def test_select_from_implicit_join(self):
        query = """SELECT * FROM t1, t2"""
        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star()],
                                                   from_table=[Identifier('t1'), Identifier('t2')]))

    def test_select_from_inner_join(self):
        query = """SELECT * FROM t1 INNER JOIN t2 ON t1.x1 = t2.x2 AND t1.x2 = t2.x2"""
        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star()],
                                                   from_table=[Join(join_type='INNER JOIN',
                                                                    left=Identifier('t1'),
                                                                    right=Identifier('t2'),
                                                                    condition=
                                                                    BinaryOperation(op='AND',
                                                                                     args_=[
                                                                                         BinaryOperation(op='=',
                                                                                                         args_=(
                                                                                                             Identifier(
                                                                                                                 't1.x1'),
                                                                                                             Identifier(
                                                                                                                 't2.x2'))),
                                                                                         BinaryOperation(op='=',
                                                                                                         args_=(
                                                                                                             Identifier(
                                                                                                                 't1.x2'),
                                                                                                             Identifier(
                                                                                                                 't2.x2'))),
                                                                                     ])

                                                                    )]))

    def test_select_from_different_join_types(self):
        join_types = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']
        for join in join_types:
            query = f"""SELECT * FROM t1 {join} t2 ON t1.x1 = t2.x2"""
            assert str(parse_sql(query)) == query
            assert str(parse_sql(query)) == str(Select(targets=[Star()],
                                                       from_table=[Join(join_type=join,
                                                                        left=Identifier('t1'),
                                                                        right=Identifier('t2'),
                                                                        condition=
                                                                        BinaryOperation(op='=',
                                                                                        args_=(
                                                                                            Identifier(
                                                                                                't1.x1'),
                                                                                            Identifier(
                                                                                                't2.x2'))),

                                                                        )]))

    def test_select_from_subquery(self):
        query = f"""SELECT * FROM (SELECT column1 FROM t1) as sub"""
        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star()],
                                                   from_table=[
                                                       Select(targets=[Identifier('column1')],
                                                              from_table=[Identifier('t1')],
                                                              alias='sub'),
                                                   ]))

    def test_select_subquery_target(self):
        query = f"""SELECT *, (SELECT 1) FROM t1"""
        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star(), Select(targets=[Constant(1)])],
                                                   from_table=[Identifier('t1')]))

        query = f"""SELECT *, (SELECT 1) as ones FROM t1"""
        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star(), Select(targets=[Constant(1)], alias='ones')],
                                                   from_table=[Identifier('t1')]))

    def test_select_subquery_where(self):
        query = f"""SELECT * WHERE column1 IN (SELECT column2 FROM t2)"""
        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star()],
                                                   where=InOperation(args_=(
                                                       Identifier('column1'),
                                                       Select(targets=[Identifier('column2')],
                                                              from_table=[Identifier('t2')])
                                                   ))))

    def test_multiple_selects(self):
        query = f"""SELECT 1; SELECT 2"""
        with pytest.raises(SQLParsingException):
            parse_sql(query)

    def test_unary_operations(self):
        unary_operations = ['-', 'NOT', ]
        for op in unary_operations:
            query = f"""SELECT {op} column1"""
            assert str(parse_sql(query)) == query
            assert str(parse_sql(query)) == str(Select(targets=[UnaryOperation(op=op, args_=(Identifier("column1"), ))],))

    def test_binary_operations(self):
        unary_operations = ['AND', 'OR', '=', '<>',  '-', '+', '*', '/', '%', '^', '<', '>', '>=', '<=',]
        for op in unary_operations:
            query = f"""SELECT column1 {op} column2"""
            assert str(parse_sql(query)) == query
            assert str(parse_sql(query)) == str(Select(targets=[BinaryOperation(op=op, args_=(Identifier("column1"), Identifier("column2")))],))

    def test_unary_comparison_predicates(self):
        ops = ['IS NULL', 'IS NOT NULL', 'IS TRUE', 'IS FALSE']
        for op in ops:
            query = f"""SELECT column1 {op}"""
            assert str(parse_sql(query)) == query
            assert str(parse_sql(query)) == str(Select(targets=[ComparisonPredicate(op=op, args_=(Identifier("column1"),))],))

    def test_functions(self):
        functions = ['max', 'min', 'avg', 'sum']
        for op in functions:
            query = f"""SELECT {op}(column1)"""
            assert str(parse_sql(query)) == query
            assert str(parse_sql(query)) == str(
                Select(targets=[Function(op=op, args_=(Identifier("column1"),))]))