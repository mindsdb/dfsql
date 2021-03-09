class pdsqlException(Exception):
    pass


class SQLParsingException(pdsqlException):
    pass


class CommandException(pdsqlException):
    pass


class QueryExecutionException(pdsqlException):
    pass

