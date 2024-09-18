import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from leadgpt.agent.lead_agent_api import LeadGPTAPI
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize LeadGPT
llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash")
lead = LeadGPTAPI(
    llm=llm,
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
        response = await lead.agent_step()
        print("Raw response from agent:", response) 
        # Parse the JSON response
        parsed_response = json.loads(response)
        return {"response": parsed_response}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from agent")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)