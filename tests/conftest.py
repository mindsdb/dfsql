import pytest
import pandas as pd
import requests
import os


@pytest.fixture()
def csv_file(tmpdir):
    # Titanic dataset first 10 lines of train
    p = tmpdir.join('titanic.csv')
    content = """passenger_id,survived,p_class,name,sex,age,sib_sp,parch,ticket,fare,cabin,embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S
    """
    p.write_text(content, encoding='utf-8')
    return p


@pytest.fixture()
def config(monkeypatch):
    from dfsql.config import Configuration

    class TestConfig(Configuration):
        pass

    TestConfig.USE_MODIN = True

    monkeypatch.setattr('dfsql.config.Configuration', TestConfig)
    return TestConfig


@pytest.fixture()
def data_source(config, csv_file, tmpdir):
    from dfsql import DataSource
    dir_path = str(csv_file.dirpath())
    ds = DataSource.from_dir(metadata_dir=str(tmpdir), files_dir_path=dir_path)
    return ds


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)


@pytest.fixture(scope='module')
def googleplay_csv(root_directory):
    path = os.path.join(root_directory, 'tests', 'googleplaystore.csv')
    url = 'https://raw.githubusercontent.com/jasonchang0/kaggle-google-apps/master/google-play-store-apps/googleplaystore.csv'

    if not os.path.exists(path):
        req = requests.get(url)
        url_content = req.content
        csv_file = open(path, 'wb')

        csv_file.write(url_content)
        csv_file.close()

    return path
