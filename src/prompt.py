from pydantic import BaseModel, Field


class Prompt(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)
