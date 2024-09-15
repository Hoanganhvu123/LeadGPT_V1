from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.litellm import ChatLiteLLM
from leadgpt.assistant.prompt import (LEAD_CONSERVATION_ASSISTANT_PROMPT,
                              STAGE_ANALYZER_ASSISTANT_PROMPT)


class StageAnalyzerAssistant(LLMChain): 
    """Assistant to analyze which conversation stage should the conversation move into."""
    @classmethod
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        template_prompt = STAGE_ANALYZER_ASSISTANT_PROMPT
        prompt = PromptTemplate(
            template = template_prompt,
            input_variables = [
                "conversation_history",
                "current_stage_id",
                "conversation_stages",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)



class LeadConversationAssistant(LLMChain):
    """Assistant to generate the next utterance for the conversation. 
       Test when without tools"""
    @classmethod
    def from_llm(
        cls,
        llm: ChatLiteLLM,
        verbose: bool = True,
        use_custom_prompt: bool = False,
        custom_prompt: str = "You are an AI Sales agent, sell me this pencil",
    ) -> LLMChain:
        """Get the response parser."""
        template_prompt = LEAD_CONSERVATION_ASSISTANT_PROMPT
        prompt = PromptTemplate(
            template=template_prompt ,
            input_variables=[
                "lead_name",
                "lead_role",
                "company_name",
                "company_business",
                "company_values",
                "conversation_purpose",
                "conversation_type",
                "current_stage",
                "conversation_history",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
