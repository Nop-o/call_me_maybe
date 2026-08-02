from typing import Any
from .function import FunctionDetails
from .prompt import Prompt
from typing import Any
import json


class JsonParsing:
    def __init__(self, json_function_details_file: str,
                 json_prompt_file: str,) -> None:
        self.json_function_details_file: Any = JsonParsing._get_json_content(
            json_function_details_file)
        self.json_prompt_file: Any = JsonParsing._get_json_content(
            json_prompt_file)
        self.functions: list[FunctionDetails] = self._create_functions()
        self.prompts: list[Prompt] = self._create_prompts()

    @staticmethod
    def _verify_duplicate_keys(data: list[tuple[Any, Any]]) -> Any:
        """Verify duplicate keys"""
        registered_keys = {}

        for key, val in data:
            if key in registered_keys:
                raise ValueError("Key error: duplicate keys entered")
            registered_keys[key] = val

        return registered_keys

    @staticmethod
    def _get_json_content(file_name: str) -> Any:
        """Open the json file and retriewe it's content"""
        with open(file_name, "r") as file:
            file_content = json.load(file, object_pairs_hook=JsonParsing._verify_duplicate_keys)
        return file_content

    def _create_functions(self) -> list[FunctionDetails]:
        """Create functions from the json file"""
        function_list: list[FunctionDetails] = []

        for data in self.json_function_details_file:
            function_list.append(FunctionDetails(**data))

        return function_list

    def _create_prompts(self) -> list[Prompt]:
        """Create prompts from the json file"""
        prompt_list: list[Prompt] = []

        for data in self.json_prompt_file:
            prompt_list.append(Prompt(**data))

        return prompt_list
