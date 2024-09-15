import json
import re
from typing import Any, Dict, List, Optional
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import CallbackManagerForChainRun
import time

class CustomAgentExecutor(AgentExecutor):
    def _parse_llm_output(self, llm_output: str) -> Dict[str, Any]:
        """Parse the LLM output and extract the JSON content."""
        try:
            # Tìm và trích xuất nội dung JSON từ đầu ra của LLM
            json_match = re.search(r'```json\n(.*?)\n```', llm_output, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
                return json.loads(json_content)
            else:
                raise ValueError("No JSON content found in LLM output")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in LLM output")

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, Any],
        inputs: Dict[str, str],
        intermediate_steps: List[tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Take a single step in the thought-action-observation loop."""
        try:
            # Get action from agent
            llm_output = self.agent.plan(
                intermediate_steps,
                **inputs,
                run_manager=run_manager,
            )
            
            # Parse LLM output
            parsed_output = self._parse_llm_output(llm_output)
            
            # Log the tool usage
            if parsed_output.get('tool'):
                print(f"Tool used: {parsed_output['tool']}")
                print(f"Tool input: {parsed_output['tool_input']}")
                
                # Execute the tool
                tool = name_to_tool_map[parsed_output['tool']]
                tool_output = tool(parsed_output['tool_input'])
                
                # Update the parsed output with the tool's result
                parsed_output['action_output'] = tool_output
            
            return parsed_output
            
        except Exception as e:
            # Handle any unexpected errors
            return {
                "conversational_stage": "Error",
                "tool": None,
                "tool_input": None,
                "action_output": None,
                "action_input": None,
                "response": f"An error occurred: {str(e)}. Let's try to continue our conversation."
            }

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run text through and get agent response."""
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        intermediate_steps: List[tuple[AgentAction, str]] = []
        start_time = time.time()
        iterations = 0
        
        while self._should_continue(iterations, time.time() - start_time):
            step_output = self._take_next_step(
                name_to_tool_map,
                inputs,
                intermediate_steps,
            )
            
            # Return the formatted agent result
            return step_output
            
        # If we get here, we've exceeded the max iterations or time limit
        return {
            "conversational_stage": "Max Iterations or Time Limit Reached",
            "tool": None,
            "tool_input": None,
            "action_output": None,
            "action_input": None,
            "response": "I apologize, but I've reached the maximum number of steps or time limit. Let me summarize our conversation so far."
        }

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        return self._call(inputs)