import json
import re
from typing import Any, Dict, List, Union

def parse_agent_result(result: Dict[str, Any], customer_information: Dict[str, Any], current_stage_id: str, current_conversation_stage: str) -> str:
    # Chuyển đổi result thành dạng có thể serialize
    serializable_result = make_serializable(result)
    
    log = serializable_result.get('log', '')
    output = serializable_result.get('output', '')

    # Phân tích log
    thoughts = re.findall(r'Thought: (.+?)(?:\n|$)', log)
    actions = re.findall(r'Action: (.+?)(?:\n|$)', log)
    action_inputs = re.findall(r'Action Input: (.+?)(?:\n|$)', log)
    observations = re.findall(r'Observation: (.+?)(?:\n|$)', log)

    # Phân tích output
    final_thought = re.search(r'Thought: (.+?)(?:\n|$)', output)
    final_response = re.search(r'DaisyBot: (.+?)(?:\n|$)', output)

    parsed_result = {
        'current_stage_id': current_stage_id,
        'current_conversation_stage': current_conversation_stage,
        'customer_information': make_serializable(customer_information),
        'thoughts': thoughts,
        'actions': actions,
        'action_inputs': action_inputs,
        'observations': observations,
        'final_thought': final_thought.group(1) if final_thought else None,
        'final_response': final_response.group(1) if final_response else None,
    }

    # Format kết quả
    formatted_result = json.dumps(parsed_result, indent=2, ensure_ascii=False)

    return formatted_result

def make_serializable(obj: Any, depth: int = 0, max_depth: int = 10) -> Union[Dict, List, str]:
    if depth > max_depth:
        return str(obj)
    
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: make_serializable(v, depth + 1, max_depth) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item, depth + 1, max_depth) for item in obj]
    elif hasattr(obj, '__dict__'):
        return make_serializable(obj.__dict__, depth + 1, max_depth)
    else:
        return str(obj)