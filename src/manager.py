from .prompt import Prompt
from .function import FunctionDetails


class Manager:
    def __init__(self, prompts: list[Prompt],
                 functions: list[FunctionDetails]) -> None:
        self.prompts = prompts
        self.functions = functions

        self.model = Small_LLM_Model()
        self.logits = self._init_logits()

    def _init_logits(self) -> None:
        """Init every logit to -inf"""
        logit_count = len(model.get_logits_from_input_ids(
            [model._tokenizer.bos_token_id]))
        self.logits = [-float('inf')] * logit_count