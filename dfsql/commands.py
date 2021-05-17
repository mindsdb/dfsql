from dfsql.engine import pd
import re
from dfsql.exceptions import CommandException


class Command:
    name = None
    default_args = {}

    def __init__(self, args):
        self.args = self.substitute_defaults(args)
        self.validate_args(self.args)

    def validate_args(self, args):
        pass

    def substitute_defaults(self, args):
        if args:
            for i, arg in enumerate(args):
                if arg is None and self.default_args.get(i):
                    args[i] = self.default_args[i]
        return args

    @classmethod
    def from_string(cls, text):
        return None

    def execute(self, data_source):
        pass


class CreateTableCommand(Command):
    name = 'CREATE TABLE'
    default_args = {
        1: True, # clean_data
    }

    def validate_args(self, args):
        if len(args) > 2:
            raise CommandException(f"Too many arguments for command {self.name}")

        if not isinstance(args[0], str):
            raise CommandException(f"First argument must be a file path, got instead: {args[0]}.")

        if len(args) > 1 and not isinstance(args[1], bool):
            raise CommandException(f"Second argument (clean_data) must be a boolean, got instead: {args[1]}")

    @classmethod
    def from_string(cls, text):
        if not text.startswith(cls.name):
            return None

        pattern = r'^CREATE TABLE \((\S+)(?:, ((?:True)|(?:False)))?\);?$'

        matches = re.match(pattern, text)
        if not matches:
            return None
        args = [(arg.strip(' \'\"') if arg is not None else None) for arg in matches.groups()]
        args[0] = str(args[0])
        if len(args) > 1 and args[1] is not None:
            args[1] = bool(args[1])
        return cls(args)

    def execute(self, data_source):
        fpath = self.args[0]
        clean = self.args[1]
        data_source.add_table_from_file(fpath, clean=clean)
        return 'OK'


class DropTableCommand(Command):
    name = 'DROP TABLE'

    def validate_args(self, args):
        if not isinstance(args[0], str):
            raise CommandException(f"Expected only argument for {self.name} to be a string table name, got instead: {args[0]}.")

    @classmethod
    def from_string(cls, text):
        if not text.startswith(cls.name):
            return None

        pattern = r'^DROP TABLE (\S+);?$'

        matches = re.match(pattern, text)
        if not matches:
            return None
        args = [(arg.strip(' \'\"') if arg is not None else None) for arg in matches.groups()]
        args[0] = str(args[0])
        return cls(args)

    def execute(self, data_source):
        name = self.args[0]
        data_source.drop_table(name)
        return 'OK'


class ShowTablesCommand(Command):
    name = 'SHOW TABLES'

    def validate_args(self, args):
        if args:
            raise CommandException(f"No arguments expected for command {self.name}")

    @classmethod
    def from_string(cls, text):
        if not text.startswith(cls.name):
            return None

        pattern = r'^SHOW TABLES\s*;?$'

        matches = re.match(pattern, text)
        if not matches:
            return None
        args = None
        return cls(args)

    def execute(self, data_source):
        rows = []
        for tname, table in data_source.tables.items():
            rows.append((table.name, table.fpath))
        return pd.DataFrame(rows, columns=['name', 'fpath'])


command_types = [CreateTableCommand, DropTableCommand, ShowTablesCommand]


def try_parse_command(sql_query):
    for command_type in command_types:
        command = command_type.from_string(sql_query)

        if command:
            return command
