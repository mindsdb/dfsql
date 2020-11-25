class DataskilletException(Exception):
    pass


class SQLParsingException(DataskilletException):
    pass


class CommandException(DataskilletException):
    pass


class QueryExecutionException(DataskilletException):
    pass

