import os
import json
import copy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from dotenv import load_dotenv

import autogen
from autogen.agentchat import Agent
from autogen.agentchat.contrib.agent_optimizer import AgentOptimizer
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.code_utils import extract_code
from autogen.math_utils import get_answer
from autogen.code_utils import execute_code

# Cấu hình API key cho GROQ
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Cấu hình cho các agent
config_list = [
    {
        'model': 'llama-3.2-90b-text-preview',
        'base_url': 'https://api.groq.com/openai/v1',
        'api_key': os.environ['GROQ_API_KEY']
    }
]

def is_termination_msg_mathchat(message):
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
    cb = extract_code(message)
    contain_code = False
    for c in cb:
        if c[0] == "python":
            contain_code = True
            break
    if message.rstrip().find("TERMINATE") >= 0:
        return True
    return not contain_code and get_answer(message) is not None and get_answer(message) != ""

class MathUserProxyAgent(MathUserProxyAgent):
    MAX_CONSECUTIVE_AUTO_REPLY = 15
    DEFAULT_REPLY = "Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)"
    PROMPTS = """Let's solve a math problem.
Query requirements:
You should always use the 'print' function for the output and use fractions/radical forms instead of decimals.
You can use packages like sympy to help you.
You must follow the formats below to write your code:
If some packages are missing, you could also suggest a code to install the corresponding package.

Please follow this process:
1. Solve the problem step by step (do not over-divide the steps).
2. Take out any queries that can be asked through Python code (for example, any calculations or equations that can be calculated) and functions you know in the context of this conversation.

Please
(1) do not mix suggested Python codes and function calls in one step.
(2) You MUST remember that you don't have a function named "python" available.

You must follow the formats below to write your Python code:

3. Wait for me to give the results or wait for the executed results of the function call.
4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.

Problem:
"""

    def __init__(
        self,
        name: Optional[str] = "MathChatAgent",
        is_termination_msg: Optional[Callable[[Dict], bool]] = is_termination_msg_mathchat,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        default_auto_reply: Optional[Union[str, Dict, None]] = DEFAULT_REPLY,
        max_invalid_q_per_step=3,
        **kwargs,
    ):
        super().__init__(
            name=name,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            default_auto_reply=default_auto_reply,
            max_invalid_q_per_step=max_invalid_q_per_step,
            **kwargs,
        )
        del self._reply_func_list[2]
        self.register_reply([Agent, None], MathUserProxyAgent._generate_math_reply, position=4)
        del self._reply_func_list[3]
        self.register_reply(
            trigger=autogen.ConversableAgent, reply_func=MathUserProxyAgent.generate_function_call_reply, position=3
        )
        self.register_reply(
            trigger=autogen.ConversableAgent, reply_func=MathUserProxyAgent._check_final_result, position=0
        )

        self.max_function_call_trial = 3
        self.query = None
        self.answer = None
        self.is_correct = None

    def generate_function_call_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[autogen.ConversableAgent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[Dict, None]]:
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if "function_call" in message:
            is_exec_success, func_return = self.execute_function(message["function_call"])
            if is_exec_success:
                self.max_function_call_trial = 3
                return True, func_return
            else:
                if self.max_function_call_trial == 0:
                    error_message = func_return["content"]
                    self.max_function_call_trial = 3
                    return (
                        True,
                        "The func is executed failed many times. "
                        + error_message
                        + ". Please directly reply me with TERMINATE. We need to terminate the conversation.",
                    )
                else:
                    revise_prompt = "You may make a wrong function call (It may due the arguments you provided doesn't fit the function arguments like missing required positional argument). \
                    If you think this error occurs due to you make a wrong function arguments input and you could make it success, please try to call this function again using the correct arguments. \
                    Otherwise, the error may be caused by the function itself. Please directly reply me with TERMINATE. We need to terminate the conversation. "
                    error_message = func_return["content"]
                    return True, "The func is executed failed." + error_message + revise_prompt
        return False, None

    def initiate_chat(
        self,
        recipient,
        answer: None,
        silent: Optional[bool] = False,
        **context,
    ):
        self.query = context["problem"]
        if not isinstance(answer, str):
            answer = str(answer)
            if answer.endswith(".0"):
                answer = answer[:-2]
            self._answer = answer
        else:
            self._answer = answer

        self.is_correct = None

        self._prepare_chat(recipient, True)
        error_message = None
        try:
            prompt = self.PROMPTS + context["problem"]
            self.send(prompt, recipient, silent=silent)
        except Exception as e:
            error_message = str(e)
            self.is_correct = 0
            print("error information: {}".format(error_message))

        recipient.reset()
        is_correct = copy.deepcopy(self.is_correct)
        self._reset()
        return is_correct

    def _check_final_result(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[autogen.Agent] = None,
        config: Optional[Any] = None,
    ):
        messages = messages[-1]
        if isinstance(messages, dict):
            messages = messages.get("content")
            if messages is None:
                return False, None

        cb = extract_code(messages)
        contain_code = False
        for c in cb:
            if c[0] == "python":
                contain_code = True
                break
        if not contain_code and get_answer(messages) is not None and get_answer(messages) != "":
            if get_answer(messages) == self._answer:
                self.is_correct = 1
                return True, "The result is Correct. Please reply me with TERMINATE."
            else:
                self.is_correct = 0
                return False, None
        else:
            return False, None

    def _reset(self):
        super()._reset()
        self.max_function_call_trial = 3
        self.is_correct = None
        self.query = None
        self.answer = None

# Load dataset
test_data, train_data = [], []
with open("MATH/dataset/algebra.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))
with open("MATH/dataset/train/algebra.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        train_data.append(json.loads(line))
test_data, train_data = test_data[0:10], train_data[0:10]

# Agents construction
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={"config_list": config_list},
)
user_proxy = MathUserProxyAgent(
    name="mathproxyagent",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "_output", "use_docker": False},
)

# Test without agent optimizations
sum = 0
for index, query in enumerate(test_data):
    is_correct = user_proxy.initiate_chat(recipient=assistant, answer=query["answer"], problem=query["question"])
    print(is_correct)
    sum += is_correct
success_rate_without_agent_training = sum / 10

# Agent Training
EPOCH = 10
optimizer = AgentOptimizer(max_actions_per_step=3, llm_config={"config_list": config_list})
for i in range(EPOCH):
    for index, query in enumerate(train_data):
        is_correct = user_proxy.initiate_chat(assistant, answer=query["answer"], problem=query["question"])
        history = assistant.chat_messages_for_summary(user_proxy)
        optimizer.record_one_conversation(history, is_satisfied=is_correct)
    register_for_llm, register_for_exector = optimizer.step()
    for item in register_for_llm:
        assistant.update_function_signature(**item)
    if len(register_for_exector.keys()) > 0:
        user_proxy.register_function(function_map=register_for_exector)

# Test with agent optimizations
sum = 0
for index, query in enumerate(test_data):
    is_correct = user_proxy.initiate_chat(recipient=assistant, answer=query["answer"], problem=query["question"])
    sum += is_correct
success_rate_with_agent_training = sum / 10

print("------------------------------------------------Functions learned------------------------------------------------")
for func in assistant.llm_config["functions"]:
    print(func["name"] + ": " + func["description"] + "\n")
print("------------------------------------------------Summary------------------------------------------------\n")
print("success_rate_without_agent_training: {average}%\n".format(average=success_rate_without_agent_training * 100))
print("success_rate_with_agent_training: {average}%\n".format(average=success_rate_with_agent_training * 100))