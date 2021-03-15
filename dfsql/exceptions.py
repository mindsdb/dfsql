class dfsqlException(Exception):
    pass


class SQLParsingException(dfsqlException):
    pass


class CommandException(dfsqlException):
    pass


class QueryExecutionException(dfsqlException):
    pass

