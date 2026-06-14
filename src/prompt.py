from pydantic import BaseModel, Field, model_validator


class Prompt(BaseModel):
    prompt: str = Field(min_length=4, max_length=200)

    @model_validator(mode='after')
    def _verify_words(self) -> 'Prompt':
        if not self.prompt[0].isupper():
            raise ValueError("Prompt error: the prompt doesn't start with a "
                             "capital letter")
        if ' ' not in self.prompt:
            raise ValueError("Prompt error: there is only one word")

        words: list[str] = self.prompt.split(" ")
        for word in words:
            if not word and word != '0':
                raise ValueError("Prompt error: too many spaces in a row")

        return self


Prompt.model_rebuild()
