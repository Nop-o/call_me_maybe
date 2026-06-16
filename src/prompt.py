from pydantic import BaseModel, Field, model_validator


class Prompt(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)

