from typing import Callable, List, Any, Optional
from langchain.prompts.base import StringPromptTemplate
from langchain.schema import AgentAction, BaseOutputParser

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools_getter: Callable
    output_parser: Optional[BaseOutputParser] = None  # Làm cho output_parser là tùy chọn

    def format(self, **kwargs) -> str:
        # Xử lý tools
        tools = self.tools_getter(kwargs.get("input", ""))
        kwargs["tools"] = self._format_tools(tools)
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])

        # Xử lý agent_scratchpad nếu có intermediate_steps
        if "intermediate_steps" in kwargs:
            kwargs["agent_scratchpad"] = self._format_intermediate_steps(kwargs["intermediate_steps"])
        else:
            kwargs["agent_scratchpad"] = ""

        # Format template với tất cả các kwargs
        return self.template.format(**kwargs)

    def _format_intermediate_steps(self, intermediate_steps: List[tuple]) -> str:
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        return thoughts

    def _format_tools(self, tools: List[Any]) -> str:
        return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

    @property
    def _type(self) -> str:
        return "custom_prompt_template"

    @property
    def input_variables(self) -> List[str]:
        # Lấy tất cả các biến từ template, ngoại trừ các biến đặc biệt
        special_vars = {"tools", "tool_names", "agent_scratchpad"}
        return [
            var.split("}")[0] for var in self.template.split("{") 
            if "}" in var and var.split("}")[0] not in special_vars
        ]