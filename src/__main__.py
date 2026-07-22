from .json_parsing import JsonParsing
from .llm_manager import LlmManager
from .json_maker import JsonMaker
from .flags import get_flags


def main() -> None:
    try:
        from pydantic import ValidationError

        flags = get_flags()
        parser = JsonParsing(flags.functions_definition,
                             flags.input)

        manager = LlmManager(parser.functions)

        json_maker = JsonMaker(manager, flags.output)
        json_maker.get_json_data(parser.prompts)
        json_maker.create_json_file()

    except ValidationError as e:
        print(e.errors()[0]["msg"].replace("Value error, ", ""))
        return
    except (FileNotFoundError, ImportError, OSError, KeyboardInterrupt) as e:
        print(e)
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
