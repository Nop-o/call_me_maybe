from .parsing import Parsing
from .manager import LlmManager


def main() -> None:
    try:
        from pydantic import ValidationError
        parser = Parsing("data/functions_definition.json",
                         "data/function_calling_tests.json")
        parser.create_functions()
        parser.create_prompts()

        manager = LlmManager(parser.prompts, parser.functions)
        encoded_functions = manager.encode_functions()

    except (FileNotFoundError, ImportError) as e:
        print(e)
        return
    except ValidationError as e:
        print(e.errors()[0]["msg"].replace("Value error, ", ""))
        return


if __name__ == "__main__":
    main()
    #try:
    #    main()
    #except Exception as e:
    #    print(e)
