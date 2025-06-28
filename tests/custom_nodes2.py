from wordcel.dag.nodes import Node


class ReverseStringNode(Node):
    description = "Node to reverse a string"

    def execute(self, input_data: str) -> str:
        return input_data[::-1]

    def validate_config(self) -> bool:
        return True
