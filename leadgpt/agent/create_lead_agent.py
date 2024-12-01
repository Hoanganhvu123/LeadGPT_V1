from __future__ import annotations

from typing import List, Optional, Sequence, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from leadgpt.agent.parser import LeadConvoOutputParser
from leadgpt.agent.format_log import format_lead_log_to_string


def create_lead_agent(
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate,
    *,
    stop_sequence: Union[bool, List[str]] = True,
) -> Runnable:
    """Create an agent that uses ReAct prompting.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use.
        tools_renderer: This controls how the tools are converted into a string and
            then passed into the LLM. Default is `render_text_description`.
        stop_sequence: bool or list of str.
            If True, adds a stop token of "Observation:" to avoid hallucinates.
            If False, does not add a stop token.
            If a list of str, uses the provided list as the stop tokens.

            Default is True. You may to set this to False if the LLM you are using
            does not support stop sequences.

    Returns:
        A Runnable sequence representing an agent.
    """
    if stop_sequence:
        stop = ["\nObservation"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm
    output_parser = LeadConvoOutputParser()
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_lead_log_to_string(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | output_parser
    )
    return agent



