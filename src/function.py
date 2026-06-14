from pydantic import BaseModel, Field, model_validator


class FunctionDetails(BaseModel):

    name: str = Field(min_length=4, max_length=40)
    description: str = Field(min_length=4, max_length=200)
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode='after')
    def _verify_function_name(self) -> "FunctionDetails":
        if not self.name.startswith("fn_"):
            raise ValueError("Name error: the function doesn't start "
                             f"with 'fn' {self.name[0:2]}")
        if " " in self.name:
            raise ValueError("Name error: the function has a ' ' (space)")
        return self

    @model_validator(mode='after')
    def _verify_description(self) -> 'FunctionDetails':
        if not self.description.endswith("."):
            raise ValueError("Description error: the description doesn't "
                             "end with '.'")

        if not self.description[0].isupper():
            raise ValueError("Description error: the description doesn't start"
                             f" with a capital letter {self.description[0]}")

        return self

    @model_validator(mode='after')
    def _verify_parameters(self) -> 'FunctionDetails':
        possible_types: list[str] = [
            "number", "string"
        ]

        for key, value in self.parameters.items():
            if not key:
                raise ValueError("Parameter error: empty key entered")
            if not value:
                raise ValueError("Parameter error: not parameter entered")

            for parameter_name, parameter_type in value.items():
                if not parameter_name:
                    raise ValueError("Parameter error: empty key entered")
                if not parameter_type:
                    raise ValueError("Parameter error: empty value entered")

                if parameter_name != "type":
                    raise ValueError("Parameter error: the parameter name "
                                     f"isn't 'type' ({parameter_name})")
                if parameter_type not in possible_types:
                    raise ValueError("Parameter error: the parameter type "
                                     f"isn't valid ({parameter_type})")

        return self

    @model_validator(mode='after')
    def _verify_returns(self) -> 'FunctionDetails':
        possible_types: list[str] = [
            "number", "string"
        ]
        if len(self.returns) > 1:
            raise ValueError("Return error: too many return value")

        for key, parameter_type in self.returns.items():

            if not key:
                raise ValueError("Return error: empty key entered")
            if not parameter_type:
                raise ValueError("Return error: empty value entered")

            if key != "type":
                raise ValueError("Return error: the return key isn't "
                                 f"'type' ({key})")
            if parameter_type not in possible_types:
                raise ValueError("Return error: the return type isn't "
                                 f"valid ({parameter_type})")

        return self


FunctionDetails.model_rebuild()
