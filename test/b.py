# Import các module cần thiết
import asyncio
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

from langchain_groq import ChatGroq
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Định nghĩa input cho công cụ tương tác web
class WebInteractionInput(BaseModel):
    url: str = Field(description="URL của trang web cần tương tác")
    action: str = Field(description="Hành động cần thực hiện trên trang web")

# Công cụ tương tác web sử dụng Selenium
class WebInteractionTool(BaseTool):
    name = "web_interaction"
    description = "Tương tác với trang web sử dụng Selenium"
    args_schema = WebInteractionInput

    def __init__(self):
        super().__init__()
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Chạy ẩn trình duyệt
        self._driver = webdriver.Chrome(options=chrome_options)

    def _run(self, url: str, action: str) -> str:
        try:
            self._driver.get(url)
            WebDriverWait(self._driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            if "search" in action.lower():
                search_term = action.split("search ")[-1]
                search_box = WebDriverWait(self._driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search'], input[type='text']"))
                )
                search_box.clear()
                search_box.send_keys(search_term)
                search_box.send_keys(Keys.RETURN)
                WebDriverWait(self._driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Lấy nội dung trang sau khi tương tác
            page_content = self._driver.find_element(By.TAG_NAME, "body").text
            return page_content

        except TimeoutException:
            return "Không thể tìm thấy phần tử trên trang web."
        except Exception as e:
            return f"Có lỗi xảy ra khi tương tác với trang web: {str(e)}"

    async def _arun(self, url: str, action: str) -> str:
        return await asyncio.to_thread(self._run, url, action)

    def close(self):
        if hasattr(self, '_driver'):
            self._driver.quit()

# Hàm chính để chạy toàn bộ chương trình
async def main():
    # Khởi tạo Web Interaction Tool
    web_tool = WebInteractionTool()

    # Thiết lập API key của Groq
    GROQ_API_KEY = "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"

    # Khởi tạo mô hình ChatGroq
    llm_groq = ChatGroq(
        temperature=0.3,
        model="llama-3.1-70b-versatile",
        groq_api_key=GROQ_API_KEY
    )

    # Thiết lập agent
    tools = [
        Tool(
            name="Web Interaction",
            func=web_tool._run,
            description="Useful for interacting with web pages",
            coroutine=web_tool._arun
        )
    ]

    agent = initialize_agent(
        tools,
        llm_groq,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Sử dụng agent để thực hiện tác vụ tương tác web
    result = await agent.arun("Tìm kiếm thông tin về 'iPhone 15' trên trang web thegioididong.com và cho tôi biết giá của nó")
    print("\nKết quả:\n")
    print(result)

    # Đóng trình duyệt
    web_tool.close()

# Chạy hàm chính
if __name__ == "__main__":
    asyncio.run(main())
