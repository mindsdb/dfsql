class DfsqlException(Exception):
    pass


class SQLParsingException(DfsqlException):
    pass


class CommandException(DfsqlException):
    pass


class QueryExecutionException(DfsqlException):
    pass

