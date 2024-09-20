import os
import asyncio
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from leadgpt.memory.summary import LeadSummaryMemory
from leadgpt.agent.lead_agent import LeadGPT
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


#------------------------------------------------------------------------

def main():
    llm = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-1.5-flash", streaming=True)
    llm_groq = ChatGroq(temperature=0.3, model="llama-3.1-70b-versatile")
    lead = LeadGPT(
        llm=llm_groq,
        verbose=True,
        lead_name="DaisyBot",
        lead_role="Sales Assistant",
        company_name="DaisyShop",
        company_business="Clothing retail",
        product_catalog="Men's and women's clothing, accessories",
        company_values="""Our mission is to serve customers with dedication and provide them with the best shopping experience. 
                        We strive to offer high-quality, fashionable clothing that meets our customers' needs and preferences.""",
        conversation_purpose="Provide product information and understand customer needs",
        conversation_type="Chat and messaging",
        languages="Vietnamese", 
    )
    
    while True:
        user_input = input("User: ")
        lead.human_step(user_input)
        lead.determine_conversation_stage()
        lead.update_customer_info()
        lead.agent_step()

#Run main loop
if __name__ == "__main__":
    main()