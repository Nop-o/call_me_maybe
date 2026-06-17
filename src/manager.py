from .prompt import Prompt
from .function import FunctionDetails
from llm_sdk.llm_sdk import Small_LLM_Model


class LlmManager:
    def __init__(self, prompts: list[Prompt],
                 functions: list[FunctionDetails]) -> None:
        self.prompts: list[list[int]] = self._get_encode_prompts(prompts)
        self.functions: list[list[int]] = self._get_encoded_functions(functions)
        self.global_prompt: list[int] = self._get_encoded_global_prompt()

        self.model: Small_LLM_Model = Small_LLM_Model()
        self.logits: list[float] = self._init_logits()

    def get_json(self) -> dict:
        return_value: dict = {}
        
        for prompt in self.prompts:
            answer: dict = {}
            
            to_tokenize = self.global_prompt + prompt + self.functions


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

    def _get_encoded_global_prompt(self) -> list[int]:
        """Get the encoded golbal prompt"""
        return self.encode_prompt(
             "You are a function-calling assistant. "
             "You receive a list of available functions and a user prompt. "
             "You must select the appropriate function to call and return "
             "its name with the required arguments. "
             "Do not explain yourself. Only output the function call."
             )[0].tolist()
