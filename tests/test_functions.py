import pytest
from dataskillet.engine import pd

from dataskillet.exceptions import QueryExecutionException
from dataskillet.functions import And


class TestFunctionBase:

    def test_and_modin_series(self):
        args = [
            pd.Series([True, False]),
            pd.Series([True, False]),
        ]

        And()(*args) == pd.Series([True, False]).all()

        args = [
            pd.Series([False, False]),
            pd.Series([True, False]),
        ]

        And()(*args) == pd.Series([False, False]).all()

    def test_and_modin_dataframe(self):
        args = [
            pd.DataFrame([[True, False]]),
            pd.DataFrame([[True, False]]),
        ]

        And()(*args) == pd.DataFrame([[True, False]]).all()

        args = [
            pd.DataFrame([[False, False]]),
            pd.DataFrame([[True, False]]),
        ]

        And()(*args) == pd.DataFrame([[False, False]]).all()

    def test_and_modin_bools(self):
        args = [
            True,
            False
        ]

        And()(*args) == False

        args = [
            True,
            True
        ]

        And()(*args) == True

    def test_and_modin_ints(self):
        args = [
            0,
            1
        ]

        And()(*args) == False

        args = [
            1,
            1
        ]

        And()(*args) == True

    def test_and_three_args(self):
        with pytest.raises(QueryExecutionException):
            And()([1, 1, 1])

    def test_and_invalid_ints(self):
        with pytest.raises(QueryExecutionException):
            And()([1, 2])

    def test_and_mixed_args(self):
        with pytest.raises(QueryExecutionException):
            And()([False, 1])

        with pytest.raises(QueryExecutionException):
            And()([pd.Series([False]), 1])

