from .prompt import Prompt
from .llm_manager import LlmManager
from pathlib import Path
import json


class JsonMaker:

    def __init__(self,  manager: LlmManager) -> None:
        self.manager: LlmManager = manager
        self.json_file_info: list[dict[str, str | dict[str, str]]] = []

    def create_json_file(
        self, data: dict[str, str | dict[str, str]]=None,
        file_name="data/output/function_calling_results.json") -> None:
            """Create a json file"""
            if data is None:
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
        print("[")
        for prompt in prompts:
            answer: dict[str, str | dict[str, str]] = {}

            print("\t{")

            answer["prompt"]: str = prompt.prompt
            print(f'\t\t"prompt": "{answer["prompt"]}",')

            answer["name"]: str = self.manager.get_function_name(prompt.prompt)
            print(f'\t\t"name": "{answer["name"]}",')

            answer["parameters"]: dict[str, str] = self.manager.get_parameters(
                self.manager.find_function_from_name(answer["name"]),
                prompt.prompt)
            print(f'\t\t"parameters": {", ".join(answer["parameters"])}')

            print("\t}")

            self.json_file_info.append(answer)
        print("]")
