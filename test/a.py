# Import các module cần thiết
import asyncio
import random
from typing import List, Dict, Any

# Import các module cho Playwright Browser Toolkit
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from playwright.async_api import async_playwright

# Import ChatGroq và các module cần thiết cho agent
from langchain_groq import ChatGroq
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# Hàm để mô phỏng hành vi con người
async def human_like_interaction(page):
    await page.mouse.move(random.randint(0, 500), random.randint(0, 500))
    await asyncio.sleep(random.uniform(0.5, 2))
    await page.evaluate("window.scrollBy(0, {})".format(random.randint(100, 500)))
    await asyncio.sleep(random.uniform(1, 3))

# Hàm chính để chạy toàn bộ chương trình
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()
        
        # Khởi tạo toolkit và lấy các công cụ
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()

        # Định nghĩa các công cụ cụ thể cho agent
        tools_for_agent: List[Tool] = [
            Tool(
                name="Navigate",
                func=lambda x: asyncio.run(tools[1].arun(x)),  # NavigateTool
                description="Dùng để điều hướng đến một URL cụ thể. Input là một URL hợp lệ bắt đầu bằng 'http://' hoặc 'https://'."
            ),
            Tool(
                name="Click",
                func=lambda x: asyncio.run(tools[0].arun(x)),  # ClickTool
                description="Dùng để click vào một phần tử trên trang web. Input là một dictionary với key 'selector'."
            ),
            Tool(
                name="Extract Text",
                func=lambda x: asyncio.run(tools[3].arun(x)),  # ExtractTextTool
                description="Dùng để trích xuất văn bản từ trang web hiện tại."
            ),
            Tool(
                name="Extract Hyperlinks",
                func=lambda x: asyncio.run(tools[4].arun(x)),  # ExtractHyperlinksTool
                description="Dùng để trích xuất các liên kết từ trang web hiện tại."
            ),
            Tool(
                name="Get Elements",
                func=lambda x: asyncio.run(tools[5].arun(x)),  # GetElementsTool
                description="Dùng để lấy thông tin về các phần tử trên trang web. Input là một dictionary với key 'selector'."
            ),
            Tool(
                name="Current Web Page",
                func=lambda x: asyncio.run(tools[6].arun(x)),  # CurrentWebPageTool
                description="Dùng để lấy thông tin về trang web hiện tại, bao gồm URL và tiêu đề."
            ),
        ]

        # Thiết lập API key của Groq
        GROQ_API_KEY = "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"

        # Khởi tạo mô hình ChatGroq
        llm_groq = ChatGroq(
            temperature=0.3,
            model="llama-3.2-90b-text-preview",
            groq_api_key=GROQ_API_KEY
        )

        # Định nghĩa prompt template cho agent
        prompt = PromptTemplate.from_template(
            """Bạn là một AI agent thông minh được thiết kế để tìm kiếm thông tin trên web.
            Nhiệm vụ của bạn là sử dụng các công cụ được cung cấp để điều hướng, tương tác và trích xuất thông tin từ các trang web.
            Hãy suy nghĩ cẩn thận về mỗi bước và sử dụng các công cụ một cách hiệu quả để tìm kiếm thông tin chính xác.

            Công cụ có sẵn:
            {tools}

            Tên các công cụ:
            {tool_names}

            Nhiệm vụ của bạn: {input}
            
            To use a tool, please use the following format:

            ```
            Thought: Do i need to use a tool? Yes
            Action: the action to take, should be one of {tools}
            Action Input: the input to the action, always a simple string input
            Observation: the result of the action
            ```

            Hãy thực hiện từng bước một cách cẩn thận và giải thích lý do cho mỗi hành động của bạn.
            Nếu bạn không tìm thấy thông tin cần thiết, hãy thử các phương pháp khác nhau hoặc tìm kiếm trên các trang web liên quan.

            Hãy bắt đầu!

            {agent_scratchpad}
            """
        )

        # Tạo React agent
        agent = create_react_agent(llm_groq, tools_for_agent, prompt)

        # Tạo Agent Executor với handle_parsing_errors=True
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools_for_agent, verbose=True, handle_parsing_errors=True
        )

        # Sử dụng agent để thực hiện nhiệm vụ
        task = "Tìm giá của iPhone 15 trên trang web thegioididong.com và cung cấp thông tin chi tiết về sản phẩm."
        try:
            # Điều hướng đến trang web trước khi thực hiện nhiệm vụ
            await page.goto("https://www.thegioididong.com")
            await human_like_interaction(page)

            result = await agent_executor.ainvoke({"input": task})
            print("\nKết quả:\n")
            print(result["output"])
        except Exception as e:
            print(f"Đã xảy ra lỗi: {str(e)}")

        # Đóng trình duyệt sau khi hoàn thành
        await browser.close()

# Chạy hàm chính
if __name__ == "__main__":
    asyncio.run(main())
