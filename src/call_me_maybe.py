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

        json_file = manager.get_json()
        # manager.create_json_file(json_file)

    except (FileNotFoundError, ImportError) as e:
        print(e)
        return
    except ValidationError as e:
        print(e.errors()[0]["msg"].replace("Value error, ", ""))
        return
    except OSError as e:
        print(e)


if __name__ == "__main__":
    main()

    #try:
    #    main()
    #except Exception as e:
    #    print(e)
