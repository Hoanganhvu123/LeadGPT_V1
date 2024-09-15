import os
from typing import Dict, Any
import json
from pprint import pprint

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables.base import RunnableSequence

from leadgpt.assistant.lead_assistant import StageAnalyzerAssistant
from leadgpt.memory.summary import LeadSummaryMemory
from leadgpt.stage import LEAD_CONVERSATION_STAGES
from leadgpt.agent.prompt import LEAD_AGENT_PROMPT
from leadgpt.tools.policy_search import policy_search_tool
from leadgpt.tools.product_search import product_search_tool


class LeadGPT:
    def __init__(self, llm, verbose=False, **kwargs):
        self.llm = llm
        self.verbose = verbose
        
        self.current_stage_id = None 
        self.current_conversation_stage = None
        self.stage_analyzer_assistant = StageAnalyzerAssistant.from_llm(llm, verbose=verbose)
        
        self.chat_memory = ChatMessageHistory()
        self.human_chat_memory = []
        self.tools = [product_search_tool, policy_search_tool]  
        self.lead_summary_memory = LeadSummaryMemory.from_messages(
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
        self.language = kwargs.get("language", "language")
        
    @property
    def current_stage(self):
        return LEAD_CONVERSATION_STAGES.get(self.current_stage_id, "Unknown stage")
     
    def determine_conversation_stage(self):   
        stage_analyzer_output = self.stage_analyzer_assistant.invoke(   
            input={
                "current_stage": self.current_stage,
                "conversation_history": self.chat_memory.messages[-6:],
                "current_stage_id": self.current_stage_id,
                "customer_information": self.lead_summary_memory.buffer
            },
            return_only_outputs=False,
        )
        self.current_stage_id = stage_analyzer_output.get("text")
        print(self.current_stage_id)

    def human_step(self) -> str:
        human_input = input("User: ")
        self.chat_memory.add_user_message(human_input)
 
    async def agent_step(self):
        inputs = {
            "leadAI_name": "LeadGPT",  
            "leadAI_role": self.lead_role,
            "company_name": self.company_name,
            "company_business": self.company_business,
            "customer_info_name": self.lead_name,
            "product_catalog": self.product_catalog,
            "company_values": self.company_values,
            "conversation_purpose": self.conversation_purpose,
            "conversation_type": self.conversation_type,
            "languages": self.language,
            "input": self.chat_memory.messages[-1].content if self.chat_memory.messages else "",
            "conversation_history": "\n".join(f"{msg.type}: {msg.content}" for msg in self.chat_memory.messages[-6:]),
            "agent_scratchpad": "",
            "customer_information_summary": self.lead_summary_memory.buffer,
            "current_stage": self.current_stage
        }
        self.lead_agent = create_react_agent(self.llm, self.tools, LEAD_AGENT_PROMPT)
        agent_executor = AgentExecutor(
            agent=self.lead_agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=4,
            return_intermediate_steps=True
        )
        result = agent_executor.invoke(inputs)
        intermediate_steps = []
        for action, observation in result['intermediate_steps']:
            intermediate_steps.append({
                "tool": action.tool,
                "input": action.tool_input,
                "output": observation
            })
        self.chat_memory.add_ai_message(result['output'])

        response = {
            "current_conversation_stage": self.current_stage,
            "current_conversation_stage_id": self.current_stage_id,
            "steps": intermediate_steps,
            "final_response": result['output'],
            "customer_information_summary": self.lead_summary_memory.buffer,
            "thought": result.get('intermediate_steps', [])[-1][0].log if result.get('intermediate_steps') else result.get('thought')
        }
        print("AI:")
        pprint(response, indent=2)  

        return response
    
    async def update_customer_infor(self):
        """Update the customer information based on summary memory usage"""
        print(f"\nPrevious_summary: {self.lead_summary_memory.buffer}\n")
        self.lead_summary_memory.buffer = self.lead_summary_memory.predict_new_summary(
            self.chat_memory.messages[-4:],
            self.lead_summary_memory.buffer
        )                                                                                                                                          
        print(f"Current Summary: {self.lead_summary_memory.buffer}\n")
