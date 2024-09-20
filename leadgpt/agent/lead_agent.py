import os
from typing import Dict, Any
import json
from pprint import pprint

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents.agent import AgentExecutor

from leadgpt.assistant.lead_assistant import StageAnalyzerAssistant
from leadgpt.memory.summary import LeadSummaryMemory
from leadgpt.stage import LEAD_CONVERSATION_STAGES
from leadgpt.agent.prompt import LEAD_AGENT_PROMPT
from leadgpt.tools.policy_search import policy_search_tool
from leadgpt.tools.product_search import product_search_tool
from leadgpt.agent.tool_prompt import CustomPromptTemplate
from leadgpt.agent.excutor import CustomAgentExecutor
from leadgpt.agent.create_lead_agent import create_lead_agent
from leadgpt.agent.result_parser import parse_agent_result

class LeadGPT:
    def __init__(self, llm, verbose=False, **kwargs):
        self.llm = llm
        self.verbose = verbose
        
        self.current_stage_id = "1"  # Giữ nguyên là chuỗi
        self.stage_analyzer_assistant = StageAnalyzerAssistant.from_llm(llm, verbose=verbose)
        
        self.chat_memory = ChatMessageHistory()
        self.human_chat_memory = []
        self.human_input = ""
        self.tools = [product_search_tool, policy_search_tool]  
        self.lead_summary_memory = LeadSummaryMemory(
            llm=llm, 
            chat_memory=self.chat_memory,
            verbose=verbose
        )
        
        self.lead_name = kwargs.get("lead_name", "lead_name")
        self.lead_role = kwargs.get("lead_role", "lead_role")
        self.company_name = kwargs.get("company_name", "company_name")
        self.company_business = kwargs.get("company_business", "company_business")
        self.product_catalog = kwargs.get("product_catalog", "product_catalog")
        self.company_values = kwargs.get("company_values", "company_values")
        self.conversation_purpose = kwargs.get("conversation_purpose", "conversation_purpose")
        self.conversation_type = kwargs.get("conversation_type", "conversation_type")
        self.languages = kwargs.get("languages", "Vietnamese")
        self.customer_info = None
        
        
    @property
    def current_conversation_stage(self):
        return LEAD_CONVERSATION_STAGES[self.current_stage_id.strip()]


    def human_step(self, message: str):
        self.human_input = message
        self.chat_memory.add_user_message(self.human_input)
        
        
    def determine_conversation_stage(self):   
        stage_analyzer_output = self.stage_analyzer_assistant.invoke(   
            input={
                "current_conversation_stage": self.current_conversation_stage,
                "conversation_history": self.chat_memory.messages[-10:],
                "current_stage_id": self.current_stage_id,
                "customer_information": self.lead_summary_memory.buffer
            },
            return_only_outputs=True,
        )
        self.current_stage_id = stage_analyzer_output.get("text", "1").strip()  
        print(f"Current Conversation Stage: {self.current_stage_id} : {self.current_conversation_stage}")
        return self.current_conversation_stage
    
    def update_customer_info(self):
        """Update the customer information based on recent conversation"""
        print(f"\nPrevious customer info: {self.customer_info}\n")
        new_info = self.lead_summary_memory.predict_new_summary(
            input={
                "customer_info": self.customer_info,
                "new_lines": [msg.content for msg in self.chat_memory.messages if msg.type == "human"][-4:]
            }
        )
        self.customer_info = new_info
        print(f"Updated customer info: {self.customer_info}\n")


    def _prepare_inputs(self) -> Dict[str, Any]:
        return {
            "leadAI_name": self.lead_name,
            "leadAI_role": self.lead_role,
            "company_name": self.company_name,
            "company_business": self.company_business,
            "company_values": self.company_values,
            "product_catalog": self.product_catalog,
            "conversation_purpose": self.conversation_purpose,
            "conversation_type": self.conversation_type,
            "languages": self.languages,  # Đổi "language" thành "languages"
            "input": self.human_input,
            "conversation_history": "\n".join(f"{msg.type}: {msg.content}" for msg in self.chat_memory.messages[-6:]),
            "customer_information": self.lead_summary_memory.buffer,
            "current_conversation_stage": self.current_conversation_stage,
            "customer_info_name": self.customer_info
        }
 
    def agent_step(self):
        inputs = self._prepare_inputs()
        prompt = CustomPromptTemplate(
            template=LEAD_AGENT_PROMPT,
            tools_getter=lambda x: self.tools,
            input_variables=[
                "input",
                "intermediate_steps",
                "leadAI_name",
                "leadAI_role",
                "company_name",
                "company_business",
                "company_values",
                "product_catalog",
                "conversation_purpose",
                "conversation_type",
                "languages",
                "conversation_history",
                "customer_information",
                "current_conversation_stage",
                "customer_info_name" 
            ],
        )
        self.runable_lead_agent = create_lead_agent(self.llm, self.tools, prompt)   
        lead_agent = CustomAgentExecutor(
            agent=self.runable_lead_agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=3,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        result = lead_agent.invoke(inputs)
        self.chat_memory.add_ai_message(result.get("output"))
        
        parsed_result = parse_agent_result(result, self.customer_info, self.current_stage_id, self.current_conversation_stage)
        print(parsed_result)
        return parsed_result

    
    