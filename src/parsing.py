from typing import Any
from .function import FunctionDetails
from .prompt import Prompt
import json


class Parsing:
    def __init__(self, json_function_details_file: str,
                 json_prompt_file: str,) -> None:
        self.json_function_details_file: Any = Parsing._get_json_content(
            json_function_details_file)
        self.json_prompt_file: Any = Parsing._get_json_content(
            json_prompt_file)
        self.functions: list[FunctionDetails] = []
        self.prompts: list[Prompt] = []

    @staticmethod
    def _get_json_content(file_name: str) -> Any:
        """Open the json file and retriewe it's content"""
        with open(file_name, "r", encoding="utf-8") as file:
            file_content = json.load(file)
        return file_content

    def create_functions(self) -> None:
        """Create functions from the json file"""
        for data in self.json_function_details_file:
            self.functions.append(FunctionDetails(**data))

    def create_prompts(self) -> None:
        """Create prompts from the json file"""
        for data in self.json_prompt_file:
            self.prompts.append(Prompt(**data))


def main() -> None:
    try:
        from pydantic import ValidationError
        parser = Parsing("data/functions_definition.json",
                         "data/function_calling_tests.json")
        parser.create_functions()
    except (FileNotFoundError, ImportError) as e:
        print(e)
        return
    except ValidationError as e:
        print(e.errors()[0]["msg"].replace("Value error, ", ""))
        return
    print(parser.functions)


if __name__ == "__main__":
    main()
