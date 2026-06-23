from .prompt import Prompt
from .llm_manager import LlmManager
from pathlib import Path
from typing import Any
import json


class JsonMaker:

    def __init__(self,  manager: LlmManager) -> None:
        self.manager: LlmManager = manager
        self.json_file_info: list[dict[str, Any]] = []

    def create_json_file(
       self, data: list[dict[str, Any]] = [],
       file_name: str = "data/output/function_calling_results.json") -> None:
        """Create a json file"""
        if not data:
            data = self.json_file_info

        try:
            directory_path = Path.cwd() / "data/output"
            directory_path.mkdir()
        except FileExistsError:
            pass

        try:
            with open(file_name, 'w') as file:
                file.write(json.dumps(data, indent=4))
        except PermissionError as e:
            print(e)

    def get_json_data(self, prompts: list[Prompt]) -> None:
        """Get json information based on given prompts"""
        # print("[")
        for prompt in prompts:
            answer: dict[str, Any] = {}

            # print("\t{")

            answer["prompt"] = prompt.prompt
            # print(f'\t\t"prompt": "{answer["prompt"]}",')

            answer["name"] = self.manager.get_function_name(prompt.prompt)
            # print(f'\t\t"name": "{answer["name"]}",')

            answer["parameters"] = self.manager.get_parameters(
                self.manager.find_function_from_name(answer["name"]),
                prompt.prompt)
            # print(f'\t\t"parameters": {", ".join(answer["parameters"])}')

            # print("\t}")

            self.json_file_info.append(answer)
        # print("]")
