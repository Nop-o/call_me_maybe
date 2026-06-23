from .prompt import Prompt
from .function import FunctionDetails
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
import json
from typing import Any


class LlmManager:
    def __init__(self, prompts: list[Prompt],
                 functions: list[FunctionDetails]) -> None:
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.constrained_logits: list[float] = self._init_logits()
        self.stop_characteres: set[int] = self._get_stop_char()
        self.prompts: list[Prompt] = prompts

        self.functions: list[FunctionDetails] = functions
        self.functions_names: set[str] = self._get_functions_names()
        self.encoded_function_names: list[
            int] = self._get_encoded_function_names()

    def get_json(self) -> list[dict[str, str | dict[str, str]]]:
        return_value: list[dict[str, str | dict[str, str]]] = []

        for prompt in self.prompts:
            answer: dict[str, str | dict[str, str]] = {}

            answer["prompt"]: str = prompt.prompt
            answer["name"]: str = self.get_function_name(prompt.prompt)
            answer["parameters"]: dict[str, str] = self.get_parameters(
                self.find_function_from_name(answer["name"]), prompt.prompt)

            return_value.append(answer)
        print(return_value)
        return return_value

    def get_function_name(self, prompt: str) -> str:
        encoded_prompt: list[int] = self.encode_prompt(
            self.get_function_information(), prompt, 'name: "')
        constrained_logits: list[float] = self.adapt_logits(
            self.encoded_function_names)

        return self.get_llm_answer(constrained_logits, encoded_prompt)

    def get_parameters(
       self, function: FunctionDetails, prompt: str) -> dict[dict[str, Any]]:
        """Get the parameters of a function with a llm"""
        parameters: dict[str, dict[str, str]] = {}
        parameter_count: int = len(function.parameters.values())
        parameters_type: list[dict[str, str]] = [
            {name: parameter['type']} for name, parameter
            in function.parameters.items()
        ]

        for i in range(parameter_count):
            (parameter_name, parameter_type), = parameters_type[i].items()
            parameters.update(self.get_parameter(
                parameters, function, parameter_name, parameter_type, prompt))

        return parameters

    def get_parameter(
       self, parameters: dict[str, dict[str, str]], function: FunctionDetails,
       parameter_name: str, parameter_type: str, prompt: str
       ) -> dict[str, Any]:
        """Get the parameter based on his type"""
        encoded_prompt: str = self.encode_prompt(
            f"name: {function.name}.\n"
            f"description: {function.description}\n",
            f"{prompt}\n",
            "parameters: {"
            f"{', '.join(f'{key}: {value}' for key, value
                         in parameters.items())
                + ', ' if parameters else ''}{parameter_name}: "
            )

        if parameter_type in ["number", "float"]:
            return {parameter_name: self._get_number_parameter(encoded_prompt)}

        elif parameter_type == "integer":
            return {parameter_name: int(self._get_number_parameter(
                encoded_prompt))}

        elif parameter_type == "string":
            return {parameter_name: self._get_string_parameter(encoded_prompt)}

        return {parameter_name: self._get_boolean_parameter(encoded_prompt)}

    def _get_number_parameter(self, encoded_prompt: str) -> dict[str, str]:
        """Get a number parameter"""
        logits: list[float] = self.adapt_logits(self.encode_list(
                ["9", "8", "7", "6", "5", "4", "3", "2", "1", "0", ".", "-"]))

        return self.get_llm_answer(logits, encoded_prompt)

    def _get_boolean_parameter(self, encoded_prompt: str) -> dict[str, str]:
        """Get a boolean parameter"""
        logits: list[
            float] = self.adapt_logits(self.encode("True") +
                                       self.encode("False"))

        return self.get_llm_answer(logits, encoded_prompt)

    def _get_string_parameter(self, encoded_prompt: str) -> dict[str, str]:
        """Get a string parameter"""
        return self.get_llm_answer([], encoded_prompt)

    def get_llm_answer(self, logits: list[float], prompt: list[int]) -> str:
        llm_answer: list[int] = []

        while 1:
            current_logits: list[
                float] = logits + self.model.get_logits_from_input_ids(prompt)

            new_token = int(np.argmax(current_logits))
            prompt.append(new_token)

            if self.is_there_a_stop_charactere(new_token):
                break
            llm_answer.append(new_token)
        print(self.decode(llm_answer))

        return self.decode(llm_answer)

    def encode_list(self, to_encode: list[str]) -> list[int]:
        encoded_list: list[int] = []

        for word in to_encode:
            encoded_list.append(int(self.model.encode(word)))

        return encoded_list

    def is_there_a_stop_charactere(self, token: int) -> bool:
        """Look if there is a stop charactere in a token"""
        decoded_token: str = self.decode(token)
        decoded_stop_characteres: str = self.decode(self.stop_characteres)

        for stop_charactere in decoded_stop_characteres:
            if stop_charactere in decoded_token:
                return True
        return False

    def create_json_file(
        self, data: dict[str, str | dict[str, str]],
        file_name="data/output/function_calling_results.json") -> None:
            with open(file_name, 'x') as file:
                file.write(json.dumps(data, indent=4))

    def find_function_from_name(
       self, function_name: str) -> FunctionDetails | None:
        """Find a FunctionDetails with it's name"""
        for function in self.functions:
            if function.name == function_name:
                return function
        return None

    def adapt_logits(self, usable_token: list[int]) -> list[float]:
        """Turn the function tokens logits to 0 instead of -inf"""
        new_logits: list[float] = self.constrained_logits.copy()

        for token in usable_token:
            new_logits[token] = 0.0
        for token in self.stop_characteres:
            new_logits[token] = 0.0

        return new_logits

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

    def encode(self, to_encode: str) -> list[int]:
        return self.model.encode(to_encode)

    def encode_prompt(
       self, system_tag: str, user_tag: str, assistant_tag: str = "") -> list[int]:
        """
        Encode a prompt with a tag system to help the llm better understand
        the received informations
        """
        return self.encode(
            '<|im_start|>system\n'
            f'{system_tag}<|im_end|>\n'
            '<|im_start|>user\n'
            f'{user_tag}<|im_end|>\n'
            '<|im_start|>assistant\n'
            f'{assistant_tag}')[0].tolist()

    def _get_functions_names(self) -> set[str]:
        """Create a set with all the functions names"""
        functions_names: set[str] = set()

        for function in self.functions:
            functions_names.add(function.name)

        return functions_names

    def get_function_information(self) -> str:
        """Encode all functions"""
        functions_informations: str = ""

        for function in self.functions:
            functions_informations += (
                f"name: {function.name}.\n"
                f"description: {function.description}\n"
                )
        return functions_informations.strip()

    def _get_stop_char(self) -> list[int]:
        list_stop_char: list[str] = [
            ",", ".", '"', " ", "\n", "<|endoftext|>"
        ]

        set_stop_char: list[int] = [
            token_id for stop_char in list_stop_char
            for token_id in self.encode(stop_char)[0].tolist()
        ]
        return set_stop_char

    def _get_encoded_function_names(self) -> list[int]:
        """Encode the functions name"""
        encoded_function_names: list[int] = []

        for function in self.functions:
            encoded_function_names.extend(
                self.encode(function.name)[0].tolist()
            )
        return encoded_function_names

    def _init_logits(self) -> list[float]:
        """Init every logit to -inf"""
        logit_count = len(self.model.get_logits_from_input_ids(
            [self.model._tokenizer.eos_token_id]))
        return np.full(logit_count, -float('inf'))
