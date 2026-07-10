from .json_parsing import JsonParsing
from .llm_manager import LlmManager
from .json_maker import JsonMaker


def main() -> None:
    try:
        from pydantic import ValidationError
        parser = JsonParsing("data/functions_definition.json",
                             "data/function_calling_tests.json")

        manager = LlmManager(parser.functions)
        json_maker = JsonMaker(manager)

        json_maker.get_json_data(parser.prompts)
        json_maker.create_json_file()

    except (FileNotFoundError, ImportError) as e:
        print(e)
        return
    except ValidationError as e:
        print(e.errors()[0]["msg"].replace("Value error, ", ""))
        return
    except OSError as e:
        print(e)
        return
    except KeyboardInterrupt as e:
        print(e)
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
