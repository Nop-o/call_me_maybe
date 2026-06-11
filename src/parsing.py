from pydantic import BaseModel


class Parsing:
    def __init__(self, file_name: str) -> None:
        self.file_content: list[str] = Parsing._get_file_content(
            file_name)

    @staticmethod
    def _get_file_content(file_name: str) -> list[str]:
        """Open the file given in argument and retriewe it's content"""
        with open(file_name, "r") as file:
            file_content = file.read()
        return file_content
    
    def create_functions(self) -> None:
        for function in self.file_content:
            




def main() -> None:
    parser = Parsing("data/functions_definition.json")
    Parsing.create_functions()

main()