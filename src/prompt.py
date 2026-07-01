from pydantic import BaseModel, Field
from llm_sdk.llm_sdk import Small_LLM_Model


class Prompt(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)


class EncodedPrompt:

    def __init__(self) -> None:
        self.model: Small_LLM_Model = Small_LLM_Model()

    def encode_prompt_to_find_function_name(
       self, system_tag: str, user_tag: str,
       assistant_tag: str = "") -> list[int]:
        """
        Encode a prompt with a tag system to help the llm better understand
        the received informations
        """
        return self.encode(
f"""
<|im_start|>system
[
    {system_tag}
]
<|im_end|>
<|im_start|>user
"{user_tag}"
<|im_end|>
<|im_start|>assistant
{"{"}
    {assistant_tag}""")

    def encode_prompt_to_find_parameters(
       self, system_tag: str, user_tag: str,
       assistant_tag: str = "", parameter_type="integer") -> list[int]:
        """
        Encode a prompt with a tag system to help the llm better understand
        the received informations
        """
        return self.encode(
f"""
<|im_start|>system
{'{'}
{system_tag}
{'}'}
<|im_end|>
<|im_start|>user
"{user_tag}"
<|im_end|>
<|im_start|>assistant
{'{'}
    {assistant_tag} {"\"" if parameter_type == "string" else ""}""")

    def decode(self, to_decode: list[int]) -> str:
        """Call the decode function from the llm"""
        return self.model.decode(to_decode)

    def encode(self, to_encode: str) -> list[int]:
        """Call the encode function from the llm"""
        return self.model.encode(to_encode)[0].tolist()
