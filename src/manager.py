from .prompt import Prompt
from .function import FunctionDetails
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np


class LlmManager:
    def __init__(self, prompts: list[Prompt],
                 functions: list[FunctionDetails]) -> None:
        self.prompts: list[list[int]] = self._get_encode_prompts(prompts)
        self.encoded_functions: list[list[int]] = self._get_encoded_functions(
            functions)
        self.global_prompt: list[int] = self._get_encoded_global_prompt()

        self.functions: list[FunctionDetails] = functions

        self.model: Small_LLM_Model = Small_LLM_Model()
        self.logits: list[float] = self._init_logits()

    def get_json(self) -> list[dict[str, str | dict[str, str]]]:
        return_value: list[dict[str, str | dict[str, str]]] = {}

        for prompt in self.prompts:
            answer: dict[str, str | dict[str, str]] = {}

            encoded_prompt = self.global_prompt + prompt + self.functions

            function_name_encoded = self.get_function_name(encoded_prompt)
            answer["name"] = self.decode(function_name_encoded)

            function_parameters_encoded = self.get_parameters(
                self.functions, answer["name"])
            answer["parameters"] = self.decode_parameters(
                function_parameters_encoded)

            return_value.append(answer)

        return return_value

    def get_function_name(self, encoded_prompt: list[int]) -> list[int]:
        function_name: str = ""

        while not self.is_str_a_function_name(function_name):
            logits = self.model.get_logits_from_input_ids(encoded_prompt)
            new_token = np.argmax(logits)
            encoded_prompt += new_token

            function_name += self.decode(new_token)

        return function_name

    def get_parameters(self, prompt: function_name) -> list[list[int]]:
        parameters: list[list[int]]  = []
        # possible_parameter_type: set[str] = set(
        #     "string", "number", "integer", "boolean"
        # )

    def get_encoded_prompt_for_parameters(
    #    self, function: FunctionDetails) -> list[int]:
    #     return self.encode_prompt(
    #         f"The function takes {len(function.parameters)} parameters.\n",
    #         "".join(
    #             [f'Parameter {i} is a {value["type"]}.\n'
    #             for i, value in enumerate(function.parameters.values())])
    #         )[0].tolist()

    def create_json_file(
       self, data: dict[str, str | dict[str, str]],
       file_name="data/output/function_calling_results.json") -> None:
        with open(file_name, 'x') as file:
            file.write(data)

    def is_str_a_function_name(self, word: str) -> bool:
        for function in self.functions:
            if function.name == word:
                return True

        return False

    def decode_parameters(
       self, encoded_parameters: list[list[int]]) -> list[str]:
        decoded_parameters: list[str] = []

        for encoded_parameter in encoded_parameters:
            decoded_parameters.append(self.decode(encoded_parameter))

        return decoded_parameters

    def decode(self, to_decode: list[int]) -> str:
        return self.model.decode(to_decode)

    def _init_logits(self) -> list[float]:

        """Init every logit to -inf"""
        logit_count = len(self.model.get_logits_from_input_ids(
            [self.model._tokenizer.eos_token_id]))
        return [-float('inf')] * logit_count

    def _get_encode_prompts(self, prompts: list[Prompt]) -> list[list[int]]:
        """Encode all prompts"""
        encoded_prompts: list[list[int]] = []

        for prompt in prompts:
            encoded_prompts.append(self.model.encode(prompt)[0].tolist())

        return encoded_prompts

    def _get_encoded_functions(self) -> list[list[int]]:
        """Encode all functions"""
        encoded_functions: list[list[int]] = []

        for function in self.functions:
            encoded_function: list[int] = self.model.encode(
                f"Function name: {function.name}\n"
                f"Function description: {function.description}\n"
                "Function parameters types: "
                f"{", ".join(value['type'] for value
                             in function.parameters.values())}."
            )[0].tolist()
            encoded_functions.append(encoded_function)

        return encoded_functions

    def _get_encoded_global_prompt(self) -> list[int]: #transformer le prompt en balise ?
        """Get the encoded golbal prompt"""
        return self.encode_prompt(
             "You are a function-calling assistant.\n"
             "You receive a list of available functions and a user prompt.\n"
             "You must select the appropriate function to call and return "
             "its name with the required arguments.\n"
             "Do not explain yourself. Only output the function call.\n"
             )[0].tolist()