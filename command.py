import argparse
from dataclasses import dataclass, fields, MISSING
from typing import Optional, Type, Any


@dataclass
class Command:
    @staticmethod
    def _get_type(field_type: Any) -> Any:
        if field_type == Optional[int]:
            return int
        elif field_type == Optional[float]:
            return float
        elif field_type == Optional[str]:
            return str
        elif field_type == Optional[list]:
            return list
        return field_type

    def parse_args(self):
        parser = argparse.ArgumentParser(description=self.__class__.__doc__)

        for field in fields(self):
            field_name = field.name
            field_type = field.type
            default = field.default

            if default == MISSING:
                # Mandatory positional arguments
                parser.add_argument(
                    field_name.replace("_", "-"), type=self._get_type(field_type)
                )
            else:
                # Optional arguments
                if field_type == bool:
                    parser.add_argument(
                        f'--{field_name.replace("_", "-")}', action="store_true"
                    )
                elif field_type == Optional[int] or field_type == int:
                    parser.add_argument(
                        f'--{field_name.replace("_", "-")}', type=int, default=default
                    )
                elif field_type == Optional[float] or field_type == float:
                    parser.add_argument(
                        f'--{field_name.replace("_", "-")}', type=float, default=default
                    )
                elif field_type == Optional[str] or field_type == str:
                    parser.add_argument(
                        f'--{field_name.replace("_", "-")}', type=str, default=default
                    )
                elif field_type == Optional[list] or field_type == list:
                    parser.add_argument(
                        f'--{field_name.replace("_", "-")}',
                        nargs="+",
                        type=str,
                        default=default,
                    )
                else:
                    raise TypeError(f"Unsupported type: {field_type}")

        args = parser.parse_args()

        # Automatically map arguments to class attributes
        for field in fields(self):
            field_name = field.name
            setattr(self, field_name, getattr(args, field_name.replace("_", "-")))

    def run(self):
        raise NotImplementedError("Subclasses should implement this method.")


@dataclass
class Repeat(Command):
    phrase: str = MISSING
    count: Optional[int] = 2
    include_counter: bool = False

    def run(self):
        repeat_count = self.count if self.count is not None else 2

        for i in range(1, repeat_count + 1):
            if self.include_counter:
                print(f"{i}: {self.phrase}")
            else:
                print(self.phrase)


@dataclass
class AnotherCommand(Command):
    message: str = MISSING
    option1: bool = False
    option2: Optional[int] = None
    values: Optional[list] = None

    def run(self):
        print(f"Message: {self.message}")
        print(f"Option1: {self.option1}")
        print(f"Option2: {self.option2 if self.option2 is not None else 'Default'}")
        if self.values:
            print(f"Values: {', '.join(self.values)}")


def main(command_class: Type[Command]):
    command = command_class()
    command.parse_args()
    command.run()


if __name__ == "__main__":
    # Example usage with Repeat class
    main(Repeat)

    # Example usage with AnotherCommand class
    # main(AnotherCommand)
