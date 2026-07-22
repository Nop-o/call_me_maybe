from .prompt import Prompt
from .llm_manager import LlmManager
from .function import FunctionDetails
from pathlib import Path
from typing import Any
import json


class JsonMaker:

    def __init__(self,  manager: LlmManager, file_path: str) -> None:
        self.manager: LlmManager = manager
        self.json_file_info: list[dict[str, Any]] = []
        self.file_path: str = file_path

    def create_json_file(
       self, data: list[dict[str, Any]] = []) -> None:
        """Create a json file"""
        if not data:
            data = self.json_file_info

        file_path = Path(self.file_path)
        if file_path.is_absolute():
            directory_path = file_path.parent
        else:
            directory_path = Path.cwd() / file_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.file_path, 'w') as file:
                file.write(json.dumps(data, indent=4))
        except PermissionError as e:
            print(e)

    def get_json_data(self, prompts: list[Prompt]) -> None:
        """Get json information based on given prompts"""
        print("[")
        for i, prompt in enumerate(prompts, 1):
            answer: dict[str, Any] = {}

            print("\t{")

            answer["prompt"] = prompt.prompt
            print(f'\t\t"prompt": "{answer["prompt"]}",')

            answer["name"] = self.manager.get_function_name(prompt.prompt)
            print(f'\t\t"name": "{answer["name"]}",')
            print('\t\t"parameters: ', end="")

            function_name: FunctionDetails | None = (
                self.manager.find_function_from_name(answer["name"]))

            if answer["name"] == "null" or function_name is None:
                answer["parameters"] = {}
                print('{}')
            else:
                answer["parameters"] = self.manager.get_parameters(
                    function_name, prompt.prompt)
                print('{')
                print(",\n".join(f'\t\t\t"{key}": "{value}"' for key, value
                                 in answer[
                    "parameters"].items()), end="")
                print('\n\t\t}')

            if i < len(prompts):
                print("\t},")
            else:
                print("\t}")

            self.json_file_info.append(answer)
        print("]")
