from typing import List, Tuple, Dict, Any, Union
from langchain_core.agents import AgentAction

def format_lead_log_to_string(intermediate_steps: List[Union[Dict[str, Any], Tuple[AgentAction, str]]]) -> str:
    """
    Format the intermediate steps of the agent's execution into a readable string.

    Args:
        intermediate_steps (List[Union[Dict[str, Any], Tuple[AgentAction, str]]]): A list of steps taken by the agent.
        Each step can be either a dictionary or a tuple containing an AgentAction and a string.

    Returns:
        str: A formatted string representing the agent's thought process and actions.

    Example:
        intermediate_steps = [
            {
                "thought": "Tôi cần tìm kiếm thông tin về sản phẩm",
                "action": "product_search_tool",
                "action_input": "áo sơ mi trắng"
            },
            (
                AgentAction(tool="product_search_tool", tool_input="áo sơ mi trắng", log="Tôi sẽ sử dụng công cụ tìm kiếm sản phẩm để tìm thông tin về áo sơ mi trắng"),
                "Tìm thấy: Áo sơ mi trắng, giá 350.000 đồng, chất liệu cotton"
            ),
            {
                "thought": "Tôi đã có thông tin về sản phẩm, giờ tôi sẽ trả lời khách hàng",
                "output": "Dạ, chúng tôi có áo sơ mi trắng với giá 350.000 đồng, được làm từ chất liệu cotton rất thoáng mát ạ."
            }
        ]
    """
    log_string = "> Entering new CustomAgentExecutor chain...\n"
    for step in intermediate_steps:
        if isinstance(step, dict):
            for key, value in step.items():
                if key == "thought" and value.startswith("Thought:"):
                    value = value.replace("Thought:", "", 1).strip()
                log_string += f"{key.capitalize()}: {value}\n"
        elif isinstance(step, tuple) and len(step) == 2:
            action, observation = step
            log_string += f"Thought: {action.log.replace('Thought:', '', 1).strip()}\n"
            log_string += f"Action: {action.tool}\n"
            log_string += f"Action Input: {action.tool_input}\n"
            log_string += f"Observation: {observation}\n"
    log_string += "> Finished chain.\n"
    # print("------------------------")
    # print()  # Thêm một dòng trống
    # print("Full log: ", log_string)
    # print()
    return log_string

