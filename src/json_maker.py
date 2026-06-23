from .prompt import Prompt
from .manager import LlmManager


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

            with open(file_name, 'x') as file:
                file.write(json.dumps(data, indent=4))

    def get_json_data(self, prompts: list[Prompt]) -> None:
        """Get json information based on given prompts"""
        for prompt in prompts:
            answer: dict[str, str | dict[str, str]] = {}

            answer["prompt"]: str = prompt.prompt
            answer["name"]: str = self.manager.get_function_name(prompt.prompt)
            answer["parameters"]: dict[str, str] = self.manager.get_parameters(
                self.manager.find_function_from_name(answer["name"]),
                prompt.prompt)

            self.json_file_info.append(answer)
