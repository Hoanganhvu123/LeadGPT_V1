from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.litellm import ChatLiteLLM
from leadgpt.assistant.prompt import STAGE_ANALYZER_ASSISTANT_PROMPT


class StageAnalyzerAssistant(LLMChain): 
    """Assistant to analyze which conversation stage should the conversation move into."""
    @classmethod
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = False) -> LLMChain:
        """Get the response parser."""
        template_prompt = STAGE_ANALYZER_ASSISTANT_PROMPT
        prompt = PromptTemplate(
            template = template_prompt,
            input_variables = [
                "conversation_history",
                "current_stage_id",
                "current_conversation_stage",
                "customer_information",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


