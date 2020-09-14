import pytest
from dataskillet.sql_parser import parse_sql, Select, Expression, Constant, Star, Identifier, BinaryOperation, FunctionCall, OrderBy


class TestParseSelect:

    def test_no_select(self):
        query = ""
        with pytest.raises(Exception):
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
                                                            FunctionCall(op="sum",
                                                                        args_=(Identifier("column3"),),
                                                                               alias='total')],
                                                   from_table=[Identifier('t1')],
                                                   group_by=[Identifier("column1"), Identifier("column2")]))

    def test_select_order_by(self):
        query = """SELECT * FROM t1 ORDER BY column1 ASC, column2, column3 DESC NULLS FIRST"""
        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star()],
                                                   from_table=[Identifier('t1')],
                                                    order_by=[
                                                        OrderBy(Identifier('column1'), direction='ASC'),
                                                        OrderBy(Identifier('column2')),
                                                        OrderBy(Identifier('column3'), direction='DESC', nulls='NULLS FIRST')],
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

    def test_select_from_inner_join(self):
        query = """SELECT * FROM t1 INNER JOIN t2 ON t1.x1 = t2.x2"""
        assert str(parse_sql(query)) == query

    def test_select_from_subquery(self):
        pass

    def test_select_subquery_target(self):
        pass

    def test_select_subquery_where(self):
        pass
