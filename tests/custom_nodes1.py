from wordcel.dag.nodes import Node


class UppercaseNode(Node):
    description = "Node to convert text to uppercase"

    def execute(self, input_data: str) -> str:
        return input_data.upper()

    def validate_config(self) -> bool:
        return True
