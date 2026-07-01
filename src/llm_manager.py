from .function import FunctionDetails
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
from typing import Any
from .prompt import EncodedPrompt


class LlmManager:
    def __init__(self, functions: list[FunctionDetails]) -> None:
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.constrained_logits: np.ndarray = self._init_logits()

        self.functions: list[
            FunctionDetails] = functions + [LlmManager._create_null_function()]
        self.encoded_function_names: list[
            int] = self._get_encoded_function_names()

        self.prompt_encoder = EncodedPrompt()

    def get_llm_answer(
       self, logits: np.ndarray, prompt: list[int],
       stop_characteres: list[int]) -> str:
        """
        Get a string from a prompt and constrained logits created by the llm
        """
        llm_answer: list[int] = []
        print(self.decode(prompt))
        while len(llm_answer) < (len(prompt) + 4):
            print(self.decode(llm_answer))
            current_logits: np.ndarray = (
                logits + self.model.get_logits_from_input_ids(prompt))

            new_token = int(np.argmax(current_logits))
            prompt.append(new_token)

            if self.is_there_a_stop_charactere(new_token, stop_characteres):
                break

            llm_answer.append(new_token)
        return self.decode(llm_answer)

    def get_function_name(self, prompt: str) -> str:
        """Get the name of a function with a llm"""
        encoded_prompt: list[
            int] = self.prompt_encoder.encode_prompt_to_find_function_name(
            self.get_function_information(), prompt, '"name": "')
        stop_characteres = self.encode_list("\"\n<|endoftext|> ")
        constrained_logits: np.ndarray = self.adapt_logits(
            self.encoded_function_names, stop_characteres)

        return self.get_llm_answer(
            constrained_logits, encoded_prompt, stop_characteres)

    def get_parameters(
       self, function: FunctionDetails, prompt: str) -> dict[str, Any]:
        """Get the parameters of a function with a llm"""
        print(function)
        print(prompt)

        if not function.parameters:
            return {}

        parameters: dict[str, Any] = {}
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
       self, parameters: dict[str, dict[str, Any]], function: FunctionDetails,
       parameter_name: str, parameter_type: str, prompt: str
       ) -> dict[str, Any]:
        """Get the parameter based on his type"""
        encoded_prompt: list[
            int] = self.prompt_encoder.encode_prompt_to_find_parameters(
            f'  "name": "{function.name}",\n'
            f'  "description": "{function.description}"',
            f'{prompt}',
            f"""
"parameters": {'{'}
        {', '.join(f'"{key}": "{value}"' for key, value in parameters.items())
         + ', ' if parameters else ''}"{parameter_name}":
            """,
            parameter_type
            )

        if parameter_type in ["string", "array"]:
            return {parameter_name: self._get_string_parameter(encoded_prompt)}

        elif parameter_type == "hexa":
            return {parameter_name: self._get_hexa_parameter(
                encoded_prompt)}

        elif parameter_type == "number":
            return {parameter_name: self._get_number_parameter(
                encoded_prompt)}

        elif parameter_type == "integer":
            return {parameter_name: int(self._get_number_parameter(
                encoded_prompt))}

        return {parameter_name: bool(
            self._get_boolean_parameter(encoded_prompt))}

    def _get_hexa_parameter(
       self, encoded_prompt: list[int]) -> float:
        """Get a hexa parameter"""
        stop_characteres = self.encode_list("\"\n<|endoftext|>")
        logits: np.ndarray = self.adapt_logits(self.encode_list(
                "9876543210abcdefABCDEF"), stop_characteres)

        return float(
            self.get_llm_answer(logits, encoded_prompt, stop_characteres))

    def _get_number_parameter(
       self, encoded_prompt: list[int]) -> float:
        """Get a number parameter"""
        stop_characteres = self.encode_list("\"\n<|endoftext|>")
        logits: np.ndarray = self.adapt_logits(self.encode_list(
                "9876543210-."), stop_characteres)

        return self.get_llm_answer(logits, encoded_prompt, stop_characteres)


    def _get_boolean_parameter(self, encoded_prompt: list[int]) -> str:
        """Get a boolean parameter"""
        stop_characteres = self.encode_list("\"\n<|endoftext|>")
        logits: np.ndarray = self.adapt_logits(self.encode("True") +
                                               self.encode("False"),
                                               stop_characteres)

        return self.get_llm_answer(logits, encoded_prompt, stop_characteres)

    def _get_string_parameter(self, encoded_prompt: list[int]) -> str:
        """Get a string parameter"""
        stop_characteres = self.encode_list("\n<|endoftext|>")

        return self.get_llm_answer(
            np.zeros(len(self.constrained_logits)),
            encoded_prompt, stop_characteres)

    def encode_list(self, to_encode: str) -> list[int]:
        """Encode a list of str into one list of token"""
        encoded_list: list[int] = []

        for word in to_encode:
            encoded_list.extend(self.encode(word))

        return encoded_list

    def is_there_a_stop_charactere(self, token: int, stop_characteres: list[int]) -> bool:
        """Look if there is a stop charactere in a token"""
        decoded_token: str = self.decode([token])
        decoded_stop_characteres: str = self.decode(stop_characteres)

        for stop_charactere in decoded_stop_characteres:
            if stop_charactere in decoded_token:
                return True
        return False

    def find_function_from_name(
       self, function_name: str) -> FunctionDetails | None:
        """Find a FunctionDetails with it's name"""
        for function in self.functions:
            if function.name == function_name:
                return function
        return None

    def adapt_logits(
       self, usable_token: list[int],
       stop_characteres : list[int]) -> np.ndarray:
        """Turn the function tokens logits to 0 instead of -inf"""
        new_logits: np.ndarray = self.constrained_logits.copy()

        for token in usable_token:
            new_logits[token] = 0.0
        for token in stop_characteres:
            new_logits[token] = 0.0

        return new_logits

    def decode(self, to_decode: list[int]) -> str:
        """Call the decode function from the llm"""
        return self.model.decode(to_decode)

    def encode(self, to_encode: str) -> list[int]:
        """Call the encode function from the llm"""
        return self.model.encode(to_encode)[0].tolist()

    def get_function_information(self) -> str:
        """Encode all functions"""
        functions_informations: str = ""

        for function in self.functions:
            functions_informations += (
    f"""
    {"{"}
        "name": "{function.name}",
        "description": "{function.description}"
    {"}"},""")
        
        return functions_informations.strip()

    def _get_encoded_function_names(self) -> list[int]:
        """Encode the functions name"""
        encoded_function_names: list[int] = []

        for function in self.functions:
            encoded_function_names.extend(
                self.encode(function.name)
            )
        encoded_function_names.extend(self.encode("null"))
        return encoded_function_names

    def _init_logits(self) -> np.ndarray:
        """Init every logit to -inf"""
        logit_count = len(self.model.get_logits_from_input_ids(
            [self.model._tokenizer.eos_token_id]))
        return np.full(logit_count, -float('inf'))

    @staticmethod
    def _create_null_function() -> FunctionDetails:
        return FunctionDetails(
            name = "null",
            description = "If you did not find any description that matches "
            "the prompt, choose this one.",
            parameters = {"null": {"type" : "null"}},
            returns = {"type": "null"}
        )
