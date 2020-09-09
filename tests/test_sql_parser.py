import pytest
from dataskillet.sql_parser import parse_sql, Select, Expression, Constant, Star, Identifier, BinaryOperation


class TestParseSelect:

    def test_no_select(self):
        query = """"""
        with pytest.raises(Exception):
            parse_sql(query)

    def test_basic_select(self):
        query = """SELECT 1, 2.0"""

        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Constant(1), Constant(2.0)]))

    def test_select_from(self):
        query = """SELECT *, column1, column1 + column2 FROM t1"""

        assert str(parse_sql(query)) == query
        assert str(parse_sql(query)) == str(Select(targets=[Star(),
                                                            Identifier("column1"),
                                                            BinaryOperation(op="+",
                                                                            args=(Identifier('column1'),
                                                                                  Identifier('column2'))
                                                                            )
                                                            ],
                                                   from_table=Identifier('t1')))

    def test_select_from_where(self):
        query = """SELECT column1, column2 FROM t1 WHERE column1 = 1"""

        assert str(parse_sql(query)) == query

        assert str(parse_sql(query)) == str(Select(targets=[Identifier("column1"), Identifier("column2")],
                                              from_table=Expression('t1'),
                                              where=BinaryOperation(op="=",
                                                                    args=(Identifier('column1'), Constant(1))
                                                                    )))
