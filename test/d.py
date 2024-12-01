import os
import random
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain_community.tools.playwright import (
    create_sync_playwright_browser,
    NavigateTool,
    ExtractTextTool,
    ExtractHyperlinksTool,
    ClickTool,
    CurrentWebPageTool
)
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

# Cấu hình API key cho GROQ
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Khởi tạo trình duyệt Playwright
sync_browser = create_sync_playwright_browser()

# Tạo các công cụ trình duyệt
navigate_tool = NavigateTool(sync_browser=sync_browser)
extract_text_tool = ExtractTextTool(sync_browser=sync_browser)
extract_hyperlinks_tool = ExtractHyperlinksTool(sync_browser=sync_browser)
click_tool = ClickTool(sync_browser=sync_browser)
current_page_tool = CurrentWebPageTool(sync_browser=sync_browser)

# Định nghĩa các công cụ
tools = [
    Tool(name="Navigate", func=navigate_tool.run, description="Navigate to a specific URL"),
    Tool(name="ExtractText", func=extract_text_tool.run, description="Extract text from the current page"),
    Tool(name="ExtractHyperlinks", func=extract_hyperlinks_tool.run, description="Extract hyperlinks from the current page"),
    Tool(name="Click", func=click_tool.run, description="Click on an element on the page"),
    Tool(name="CurrentPage", func=current_page_tool.run, description="Get information about the current page")
]

# Khởi tạo mô hình ngôn ngữ
llm = ChatGroq(
    temperature=0.3,
    model="llama-3.2-90b-text-preview",
    groq_api_key=GROQ_API_KEY
)

# Định nghĩa trạng thái
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next: str
    current_url: str
    page_content: str
    discovered_links: list

# Hàm xử lý tin nhắn của người dùng
def user(state: State, user_input: str) -> State:
    state["messages"].append(HumanMessage(content=user_input))
    return state

# Hàm xử lý phản hồi của AI
def ai(state: State) -> State:
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một trợ lý AI thông minh có khả năng tìm kiếm và phân tích thông tin trên web.
        Nhiệm vụ của bạn là:
        1. Phân tích yêu cầu tìm kiếm và xác định từ khóa.
        2. Sử dụng trình duyệt để tìm kiếm trên Google và các trang web liên quan.
        3. Đọc kết quả tìm kiếm và chọn trang web phù hợp.
        4. Truy cập vào trang web đó và tìm thông tin cần thiết.
        5. Tổng hợp thông tin và trả lời câu hỏi.
        
        Bạn phải làm việc nhanh và chính xác. Nếu không tìm thấy thông tin, hãy thử tìm kiếm lại với từ khóa khác."""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    response = llm.invoke(prompt.format_prompt(messages=messages).to_messages())
    state["messages"].append(AIMessage(content=response.content))

    # Kiểm tra xem có cần thực hiện tìm kiếm web không
    if "cần tìm kiếm" in response.content.lower() or "tìm trên web" in response.content.lower():
        state["next"] = "search_google"
    else:
        state["next"] = END
    
    return state

# Hàm tìm kiếm trên Google
def search_google(state: State) -> State:
    messages = state["messages"]
    last_ai_message = messages[-1].content
    
    # Trích xuất từ khóa tìm kiếm từ tin nhắn cuối cùng của AI
    search_query = last_ai_message.split("tìm kiếm:")[-1].split(".")[0].strip()
    
    # Thực hiện tìm kiếm trên Google
    navigate_tool.run(f"https://www.google.com/search?q={search_query}")
    
    # Đọc nội dung trang kết quả tìm kiếm
    state["page_content"] = extract_text_tool.run()
    state["current_url"] = current_page_tool.run()["url"]
    state["discovered_links"] = extract_hyperlinks_tool.run()
    
    state["messages"].append(AIMessage(content=f"Đã tìm kiếm '{search_query}' trên Google. Đang phân tích kết quả..."))
    
    state["next"] = "analyze_search_results"
    return state

# Hàm phân tích kết quả tìm kiếm và chọn trang web phù hợp
def analyze_search_results(state: State) -> State:
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một trợ lý AI thông minh có khả năng phân tích kết quả tìm kiếm.
        Nhiệm vụ của bạn là:
        1. Phân tích các kết quả tìm kiếm hiện tại.
        2. Chọn trang web phù hợp nhất với yêu cầu tìm kiếm.
        3. Đề xuất URL cần truy cập tiếp theo."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", f"Kết quả tìm kiếm:\n{state['page_content'][:1000]}...\n\nCác liên kết:\n{state['discovered_links'][:10]}\n\nBạn nghĩ chúng ta nên truy cập trang web nào tiếp theo?"),
    ])
    response = llm.invoke(prompt.format_prompt(messages=messages).to_messages())
    state["messages"].append(AIMessage(content=response.content))
    
    # Trích xuất URL cần truy cập từ phản hồi của AI
    url_to_visit = [link for link in state["discovered_links"] if link in response.content][0]
    
    state["next"] = "navigate_to_website"
    state["url_to_visit"] = url_to_visit
    return state

# Hàm điều hướng đến trang web đã chọn
def navigate_to_website(state: State) -> State:
    url_to_visit = state["url_to_visit"]
    navigate_tool.run(url_to_visit)
    
    state["current_url"] = current_page_tool.run()["url"]
    state["page_content"] = extract_text_tool.run()
    state["discovered_links"] = extract_hyperlinks_tool.run()
    
    state["messages"].append(AIMessage(content=f"Đã truy cập: {url_to_visit}. Đang phân tích nội dung..."))
    
    state["next"] = "analyze_website_content"
    return state

# Hàm phân tích nội dung trang web và tìm thông tin
def analyze_website_content(state: State) -> State:
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một trợ lý AI thông minh có khả năng phân tích nội dung trang web.
        Nhiệm vụ của bạn là:
        1. Phân tích nội dung trang web hiện tại.
        2. Tìm kiếm thông tin cần thiết.
        3. Quyết định xem có cần tìm kiếm thêm thông tin không.
        4. Nếu cần, đề xuất hành động tiếp theo (ví dụ: click vào một liên kết cụ thể).
        5. Nếu đã đủ thông tin, tổng hợp và trả lời câu hỏi."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", f"Nội dung trang web:\n{state['page_content'][:1000]}...\n\nCác liên kết:\n{state['discovered_links'][:10]}\n\nBạn đã tìm thấy thông tin cần thiết chưa? Nếu chưa, chúng ta nên làm gì tiếp theo?"),
    ])
    response = llm.invoke(prompt.format_prompt(messages=messages).to_messages())
    state["messages"].append(AIMessage(content=response.content))
    
    if "cần tìm kiếm thêm" in response.content.lower() or "click vào" in response.content.lower():
        state["next"] = "perform_action"
    else:
        state["next"] = END
    
    return state

# Hàm thực hiện hành động trên trang web (ví dụ: click vào liên kết)
def perform_action(state: State) -> State:
    messages = state["messages"]
    last_ai_message = messages[-1].content
    
    if "click vào" in last_ai_message.lower():
        # Trích xuất phần tử cần click từ tin nhắn của AI
        element_to_click = last_ai_message.split("click vào")[-1].split(".")[0].strip()
        click_tool.run(element_to_click)
        
        state["current_url"] = current_page_tool.run()["url"]
        state["page_content"] = extract_text_tool.run()
        state["discovered_links"] = extract_hyperlinks_tool.run()
        
        state["messages"].append(AIMessage(content=f"Đã click vào '{element_to_click}'. Đang phân tích nội dung mới..."))
    else:
        state["messages"].append(AIMessage(content="Không thể thực hiện hành động yêu cầu."))
    
    state["next"] = "analyze_website_content"
    return state

# Xây dựng đồ thị trạng thái
workflow = StateGraph(State)

# Thêm các nút vào đồ thị
workflow.add_node("user", user)
workflow.add_node("ai", ai)
workflow.add_node("search_google", search_google)
workflow.add_node("analyze_search_results", analyze_search_results)
workflow.add_node("navigate_to_website", navigate_to_website)
workflow.add_node("analyze_website_content", analyze_website_content)
workflow.add_node("perform_action", perform_action)

# Xác định luồng của đồ thị
workflow.set_entry_point("user")
workflow.add_edge("user", "ai")
workflow.add_edge("ai", "search_google")
workflow.add_edge("search_google", "analyze_search_results")
workflow.add_edge("analyze_search_results", "navigate_to_website")
workflow.add_edge("navigate_to_website", "analyze_website_content")
workflow.add_edge("analyze_website_content", "perform_action")
workflow.add_edge("perform_action", "analyze_website_content")
workflow.add_edge("analyze_website_content", END)

# Biên dịch đồ thị
app = workflow.compile()

# Hàm chạy cuộc trò chuyện
def run_conversation():
    state = {"messages": [], "current_url": "", "page_content": "", "discovered_links": []}
    
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == 'quit':
            break
        
        for output in app.stream(state | {"messages": state["messages"] + [HumanMessage(content=user_input)]}):
            if "ai" in output:
                print("AI:", output["ai"]["messages"][-1].content)
            elif "search_google" in output:
                print("Đang tìm kiếm trên Google...")
            elif "analyze_search_results" in output:
                print("Đang phân tích kết quả tìm kiếm...")
            elif "navigate_to_website" in output:
                print("Đang truy cập trang web...")
            elif "analyze_website_content" in output:
                print("Đang phân tích nội dung trang web...")
            elif "perform_action" in output:
                print("Đang thực hiện hành động trên trang web...")
        
        state = output

# Chạy cuộc trò chuyện
if __name__ == "__main__":
    print("Bắt đầu cuộc trò chuyện. Nhập 'quit' để kết thúc.")
    run_conversation()
    print("Cuộc trò chuyện đã kết thúc!")

# Đóng trình duyệt khi kết thúc
sync_browser.close()