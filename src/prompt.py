from pydantic import BaseModel, Field
from llm_sdk.llm_sdk import Small_LLM_Model


class Prompt(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)


class PromptEncoder:

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model: Small_LLM_Model = model

    def encode_prompt(
       self, system_tag: str, user_tag: str,
       assistant_tag: str = "", parameter_type: str = "integer") -> list[int]:
        """
        Encode a prompt with a tag system to help the llm better understand
        the received informations
        """
        return self.encode(
            f"""
<|im_start|>system
{system_tag}
<|im_end|>
<|im_start|>user
"{user_tag}"
<|im_end|>
<|im_start|>assistant
{assistant_tag}""")

    def encode(self, to_encode: str) -> list[int]:
        """Call the encode function from the llm"""
        return list(self.model.encode(to_encode)[0].tolist())

    def encode_prompt_to_find_function_name(
       self, system_tag: str, user_tag: str, assistant_tag: str) -> list[int]:
        """
        Encode a prompt with a tag system to help the llm better understand
        the received informations
        """
        return self.encode(
            '<|im_start|>system\n'
            '[\n'
            f'\t{system_tag}\n'
            ']<|im_end|>\n'
            '<|im_start|>user\n'
            f'{user_tag}'
            '<|im_end|>\n'
            '<|im_start|>assistant\n'
            '{\n'
            f'\t{assistant_tag}')

    def encode_prompt_to_find_parameters(
       self, system_tag: str, user_tag: str, assistant_tag: str) -> list[int]:
        """
        Encode a prompt with a tag system to help the llm better understand
        the received informations
        """
        return self.encode(
            '<|im_start|>system\n'
            '{\n'
            f'{system_tag}\n'
            '}<|im_end|>\n'
            '<|im_start|>user\n'
            f'{user_tag}'
            '<|im_end|>\n'
            '<|im_start|>assistant\n'
            '{\n'
            '\t"parameters": {\n'
            f'\t\t{assistant_tag}')
