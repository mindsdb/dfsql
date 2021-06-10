# dfsql - SQL interface to Pandas.

# Installation
```pip install dfsql```

# Example
```
>>> import pandas as pd
>>> from dfsql import sql_query

>>> df = pd.DataFrame({
...     "animal": ["cat", "dog", "cat", "dog"],
...     "height": [23,  100, 25, 71] 
... })
>>> df.head()
  animal  height
0    cat      23
1    dog     100
2    cat      25
3    dog      71
>>> sql_query("SELECT animal, height FROM animals_df WHERE height > 50", animals_df=df)
  animal  height
0    dog     100
1    dog      71
```

# Quickstart/Tutorial

Head over to the [testdrive notebook](https://github.com/mindsdb/dfsql/blob/stable/testdrive.ipynb) to see all available features.

# Configuring Modin usage

dfsql supports executing queries using Modin for enchanced performance. 

By default Modin will be used if it's installed.

To override this behavior and use Pandas set the `USE_MODIN` environment variable to `False` or `0` before importing dfsql:
```
(venv) user:~/mindsdb/dfsql$ export USE_MODIN=0
(venv) user:~/mindsdb/dfsql$ python
Python 3.8.5 (default, Jan 27 2021, 15:41:15) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import dfsql
>>> dfsql.config.Configuration.as_dict()
{'USE_MODIN': 0}
```



