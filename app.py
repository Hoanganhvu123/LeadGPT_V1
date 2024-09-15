import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.json_chat.base import create_json_chat_agent
from langchain.prompts import ChatPromptTemplate, SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough

from leadgpt.assistant.lead_assistant import StageAnalyzerAssistant
from leadgpt.memory.summary import LeadSummaryMemory
from leadgpt.stage import LEAD_CONVERSATION_STAGES
from leadgpt.agent.prompt import LEAD_AGENT_PROMPT
from leadgpt.tool.rag_tool import get_tools
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['SERPAPI_API_KEY'] = os.getenv('SERPAPI_API_KEY')

app = FastAPI()

class UserInput(BaseModel):
    message: str

class LeadGPT:
    def __init__(self, llm, verbose=False, **kwargs):
        self.llm = llm
        self.embeddings = OpenAIEmbeddings()
        self.verbose = verbose
        self.data_path = "E:\\LeadGPT\\packages\\leadgpt\\data\\Epacific.txt"
        self.vectorstore_path = "E:\\my-app\\packages\\leadgpt\\data\\vectorstore"
        
        self.current_stage_id = None 
        self.stage_analyzer_assistant = StageAnalyzerAssistant.from_llm(llm, verbose=verbose)
        
        self.chat_memory = ChatMessageHistory()
        self.human_chat_memory = []
        self.lead_summary_memory = LeadSummaryMemory.from_messages(llm=llm, chat_memory=self.chat_memory, verbose=verbose)
        
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
        return self.current_stage_id

    def human_step(self, human_input: str):
        self.chat_memory.add_user_message(human_input)
        
    def _prepare_inputs(self, query: str = None) -> Dict[str, Any]:
        inputs = {
            "lead_name": self.lead_name,
            "lead_role": self.lead_role,
            "company_name": self.company_name,
            "company_business": self.company_business,
            "product_catalog": self.product_catalog,
            "company_values": self.company_values,
            "conversation_purpose": self.conversation_purpose,
            "conversation_type": self.conversation_type,
            "language": self.language,
            "current_stage": self.current_stage,
            "conversation_history": self.chat_memory.messages[-6:],
            "human_input": query if query else ""
        }
        return inputs
    
    async def agent_step(self, human_input: str):
        inputs = self._prepare_inputs(human_input)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=LEAD_AGENT_PROMPT.format(**inputs)),
            HumanMessage(content="{human_input}"),
            AIMessage(content="Certainly! I'll assist you with that. Let me think about the best way to respond."),
        ])
        
        tools = get_tools()  # Assuming this function exists and returns the necessary tools
        
        agent = create_json_chat_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.verbose,
            max_iterations=2
        )
        
        agent_result = await agent_executor.ainvoke(inputs)
        
        result = {
            "conversational_stage": self.determine_conversation_stage(),
            "response": agent_result["output"],
            "tool": agent_result.get("tool"),
            "tool_input": agent_result.get("tool_input"),
            "action_output": agent_result.get("action_output"),
            "action_input": agent_result.get("action_input")
        }
        
        self.chat_memory.add_ai_message(result["response"])
        return result
    
    async def update_customer_info(self):
        print(f"Previous summary: {self.lead_summary_memory.buffer}\n")
        self.lead_summary_memory.buffer = self.lead_summary_memory.predict_new_summary(
            self.chat_memory.messages[-4:],
            self.lead_summary_memory.buffer
        )
        print(f"Current Summary: {self.lead_summary_memory.buffer}\n")



lead_gpt = LeadGPT(
    llm=ChatGroq(temperature=0, model_name="llama3-8b-8192"),
    verbose=False,
    lead_name="Epa AI",
    lead_role="Nhân viên bán hàng",
    company_name="Epacific Telecom",
    company_business="Công nghệ viễn thông",
    product_catalog="Công nghệ Ccall, công nghệ Var, công nghệ Eflowai",
    company_values="""Sứ mệnh của Công ty là trở thành một trong những nhà sản xuất và cung cấp thông tin
                    và giải trí hàng đầu thế giới. Sử dụng danh mục các thương hiệu của chúng tôi để phân biệt nội dung, 
                    dịch vụ và sản phẩm tiêu dùng của chúng tôi, chúng tôi tìm cách phát triển những trải nghiệm giải trí 
                    và sản phẩm liên quan sáng tạo, đổi mới và có lợi nhất trên thế giới""",
    conversation_purpose="Cung cấp thông tin sản phẩm, thuyết phục khách hàng mua hàng và thu thập thông tin khách hàng",
    conversation_type="Hình thức trò chuyện, nhắn tin",
    language="Vietnamese",
)


@app.post("/chat")
async def chat(user_input: UserInput):
    lead_gpt.human_step(user_input.message)
    result = await lead_gpt.agent_step(user_input.message)
    await lead_gpt.update_customer_info()
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)