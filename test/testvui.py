import asyncio
import nest_asyncio  # Import nest_asyncio
from langchain.agents import AgentType, initialize_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Initialize the LLM (Language Model)
llm = ChatGoogleGenerativeAI(
    temperature=0,
    model="gemini-1.5-flash",
    streaming=True
)

# Function to set up Playwright browser and toolkit
async def setup():
    try:
        # Create an asynchronous Playwright browser instance
        async_browser = await create_async_playwright_browser()
        # Initialize the PlayWrightBrowserToolkit with the browser
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
        # Retrieve the tools from the toolkit
        tools = toolkit.get_tools()
        return tools
    except Exception as e:
        print(f"Error during setup: {e}")
        raise

# Main asynchronous function
async def main():
    try:
        # Set up the tools
        tools = await setup()
        
        # Initialize the agent with the tools and LLM
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        # Define your query
        query = "Tìm cho tôi chiếc áo có giá rẻ nhất trên shopee"
        
        # Run the agent asynchronously
        result = await agent.arun(query)
        
        # Print the result
        print(result)
    except Exception as e:
        print(f"Error in main: {e}")

# Entry point of the script
if __name__ == "__main__":
    try:
        # Run the main function using asyncio
        asyncio.run(main())
    except Exception as e:
        print(f"Error running the script: {e}")
