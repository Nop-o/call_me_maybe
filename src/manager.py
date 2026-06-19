from .prompt import Prompt
from .function import FunctionDetails
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np


class LlmManager:
    def __init__(self, prompts: list[Prompt],
                 functions: list[FunctionDetails]) -> None:
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.constrained_logits: list[float] = self._init_logits()
        self.prompts: list[Prompt] = prompts
        self.stop_characteres: set[int] = self._get_stop_char()

        self.functions: list[FunctionDetails] = functions
        self.functions_names: set[str] = self._get_functions_names()
        self.encoded_function_names: list[
            int] = self._get_encoded_function_names()
        self.encoded_function_prompts: list[list[
            int]] = self._get_encoded_functions()

        self.global_prompt: list[int] = self._get_encoded_global_prompt()

    def get_json(self) -> list[dict[str, str | dict[str, str]]]:
        return_value: list[dict[str, str | dict[str, str]]] = {}

        for prompt in self.prompts:
            answer: dict[str, str | dict[str, str]] = {}

            answer["prompt"] = prompt.prompt
            answer["name"] = self.get_function_name(prompt.prompt)
            answer["parameters"] = self.get_parameters(
                self.find_function_from_name(answer["name"]))

            return_value.append(answer)

        return return_value

    def get_function_name(self, prompt: str) -> str:
        function_name: str = ""
        encoded_prompt: list[int] = (
                self.global_prompt
                + self.encoded_function_prompts
                + self.encode_prompt(prompt))
        print(self.model.decode(encoded_prompt))

        self.adapt_logits(self.encoded_function_names)
        new_token = None

        while new_token not in self.stop_characteres:
            logits: list[float] = self.model.get_logits_from_input_ids(
                encoded_prompt)
            logits += self.constrained_logits

            new_token = np.argmax(logits)
            encoded_prompt.append(int(new_token))

            function_name += self.decode(new_token)
            print(f"function: {function_name}",  flush=True)
        print("FOUND")

        return function_name

    def get_parameters(self, function: FunctionDetails) -> list[str]:

        parameters: list[str] = []
        parameter_count: int = len(function.parameters.values())
        parameters_type: list[str] = [
            parameter['type'] for parameter in function.parameters.values()
        ]
        encoded_prompt = self.model.encode(
            f"This function has {parameter_count} parameters.\n")

        for i in range(parameter_count):
            parameter: str = ""

            while (1):
                encoded_prompt += self.get_encoded_prompt_for_one_parameter(
                    parameters_type[i])
                logits = self.model.get_logits_from_input_ids(encoded_prompt)
                new_token = np.argmax(logits)
                encoded_prompt += new_token

                decoded_token += self.decode(new_token)
                if decoded_token == ',':
                    if parameters_type[i] == "string":
                        encoded_prompt += self.model.encode('"')
                        parameter += '"'
                    elif ((parameters_type[i] ==  "number" or
                          parameters_type[i] == "float") and
                          '.' not in parameter):
                        parameter += '.0'
                    break

                parameter += decoded_token
            parameters.append(parameter)
        
        return parameters

    def get_encoded_prompt_for_one_parameter(
       self, parameter: str) -> list[int]:
        if parameter == "number":
            return self.model.encode(
                "The next parameter is a number: "
            )[0].tolist()

        elif parameter == "float":
            return self.model.encode(
                "The next parameter is a float: "
            )[0].tolist()

        elif parameter == "integer":
            return self.model.encode(
                "The next parameter is an integer: "
            )[0].tolist()

        elif parameter == "string":
            return self.model.encode(
                'The next parameter is a string: "'
            )[0].tolist()

        else:
            return self.model.encode(
                'The next parameter is a boolean: '
            )[0].tolist()

    def create_json_file(
       self, data: dict[str, str | dict[str, str]],
       file_name="data/output/function_calling_results.json") -> None:
        with open(file_name, 'x') as file:
            file.write(data)

    def find_function_from_name(
       self, function_name: str) -> FunctionDetails | None:
        """Find a FunctionDetails with it's name"""
        for function in self.functions:
            if function.name == function_name:
                return function
        return None

    def adapt_logits(self, usable_token: list[int]) -> None:
        """Turn the function tokens logits to 0 instead of -inf"""
        print(usable_token)
        usable_token += self.stop_characteres
        print(usable_token)
        for token in usable_token:
            self.constrained_logits[token] = 0.0

    def is_str_a_function_name(self, word: str) -> bool:
        """Return True if the word is a valid function name"""
        return word in self.functions_names

    def decode_parameters(
       self, encoded_parameters: list[list[int]]) -> list[str]:
        decoded_parameters: list[str] = []

        for encoded_parameter in encoded_parameters:
            decoded_parameters.append(self.decode(encoded_parameter))

        return decoded_parameters

    def decode(self, to_decode: list[int]) -> str:
        return self.model.decode(to_decode)

    def _get_encoded_function_names(self) -> list[int]:
        """Encode the functions name"""
        encoded_function_names: list[int] = []

        for function in self.functions:
            encoded_function_names.extend(
                self.model.encode(function.name)[0].tolist()
            )
        return encoded_function_names

    def _init_logits(self) -> list[float]:
        """Init every logit to -inf"""
        logit_count = len(self.model.get_logits_from_input_ids(
            [self.model._tokenizer.eos_token_id]))
        return np.full(logit_count, -float('inf'))

    def encode_prompt(self, prompt: str) -> list[int]:
        """Encode a prompt"""
        return (self.model.encode(
                '<|im_start|>user\n'
                f'{prompt}<|im_end|>\n'
                '<|im_start|>assistant\n'
                'name: '
                )[0].tolist())

    def _get_functions_names(self) -> set[str]:
        """Create a set with all the functions names"""
        functions_names: set[str] = set()

        for function in self.functions:
            functions_names.add(function.name)

        return functions_names

    def _get_encoded_functions(self) -> list[int]:
        """Encode all functions"""
        encoded_function_prompts: list[int] = []
        # (
        #     self.model.encode("The possible function name are: ")[0].tolist())
        # encoded_function_prompts.extend(self.encoded_function_names)

        for function in self.functions:
            encoded_function: list[int] = self.model.encode(
                f"name: {function.name}.\n"
                f"description: {function.description}\n"
                )[0].tolist()
            encoded_function_prompts.extend(encoded_function)
        encoded_function_prompts.extend(self.model.encode(
            "<|im_end|>")[0].tolist())
        print(encoded_function_prompts)
        return encoded_function_prompts

    # def _get_encoded_function_names(self) -> list[int]:
    #     return (self.model.encode(f"{', '.join(function.name for function
    #             in self.functions)}.\n")[0].tolist())

    def _get_encoded_global_prompt(self) -> list[int]:
        """Get the encoded golbal prompt"""
        return self.model.encode(
            "<|im_start|>system\n"
            )[0].tolist()

    def _get_stop_char(self) -> set[int]:
        list_stop_char: list[str] = [
            ",", ".", '"', " ", "\n", "<|endoftext|>"
        ]

        set_stop_char: set[int] = {
            token_id for stop_char in list_stop_char
            for token_id in self.model.encode(stop_char)[0].tolist()
        }
        return set_stop_char
