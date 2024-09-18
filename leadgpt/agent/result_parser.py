import json
import re

def parse_agent_result(result_json):
    result = json.loads(result_json)
    log = result.get('log', '')
    output = result.get('output', '')

    # Phân tích log
    thoughts = re.findall(r'Thought: (.+?)(?:\n|$)', log)
    actions = re.findall(r'Action: (.+?)(?:\n|$)', log)
    action_inputs = re.findall(r'Action Input: (.+?)(?:\n|$)', log)
    observations = re.findall(r'Observation: (.+?)(?:\n|$)', log)

    # Phân tích output
    final_thought = re.search(r'Thought: (.+?)(?:\n|$)', output)
    final_response = re.search(r'DaisyBot: (.+?)(?:\n|$)', output)

    parsed_result = {
        'thoughts': thoughts,
        'actions': actions,
        'action_inputs': action_inputs,
        'observations': observations,
        'final_thought': final_thought.group(1) if final_thought else None,
        'final_response': final_response.group(1) if final_response else None
    }

    # Format kết quả
    formatted_result = json.dumps(parsed_result, indent=2, ensure_ascii=False)

    return formatted_result