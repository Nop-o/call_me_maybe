from .parsing import Parsing


def main() -> None:
    try:
        from pydantic import ValidationError
        parser = Parsing("data/functions_definition.json",
                         "data/function_calling_tests.json")
        parser.create_functions()
        parser.create_prompts()
        print(parser.functions)
        print()
        print(parser.prompts)
    except (FileNotFoundError, ImportError) as e:
        print(e)
        return
    except ValidationError as e:
        print(e.errors()[0]["msg"].replace("Value error, ", ""))
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
