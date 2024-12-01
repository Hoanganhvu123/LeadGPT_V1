import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from leadgpt.agent.lead_agent import LeadGPT
import json
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LeadGPT
llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash")
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

class Message(BaseModel):
    content: str

@app.post("/chat")
async def chat(message: Message):
    try:
        lead.human_step(message.content)
        lead.determine_conversation_stage()
        lead.update_customer_info()
        response = lead.agent_step()
        # Parse the JSON response
        return {"response": response}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from agent")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)