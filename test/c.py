import autogen
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

# Cấu hình API key cho GROQ (nếu cần)
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Cấu hình cho các agent
config_list = [
    {
        'model': 'llama-3.2-90b-text-preview',
        'base_url': 'https://api.groq.com/openai/v1',
        'api_key': os.environ['GROQ_API_KEY']
    }
]

# Hàm tìm kiếm sản phẩm trên Ivymoda
def search_ivymoda(max_price=500000):
    url = "https://ivymoda.com/danh-muc/hang-nu-moi-ve"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    products = []
    for item in soup.find_all('div', class_='item'):
        name = item.find('h3', class_='title').text.strip()
        price = item.find('div', class_='price').text.strip()
        price = int(''.join(filter(str.isdigit, price)))
        
        if price <= max_price:
            products.append({
                'name': name,
                'price': price
            })
    
    return products

# Cấu hình tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_ivymoda",
            "description": "Tìm kiếm sản phẩm trên Ivymoda dưới 500k",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {
                        "type": "number",
                        "description": "Giá tối đa của sản phẩm"
                    }
                },
                "required": []
            }
        }
    }
]

# Tạo agent tìm kiếm sản phẩm
ivymoda_agent = autogen.AssistantAgent(
    name="Ivymoda_Agent",
    llm_config={"config_list": config_list, "tools": tools},
    system_message="""Bạn là một trợ lý mua sắm thông minh. Nhiệm vụ của bạn là:
    1. Sử dụng hàm search_ivymoda để tìm kiếm sản phẩm dưới 500k trên Ivymoda.
    2. Phân tích và tổng hợp kết quả tìm kiếm.
    3. Đề xuất các sản phẩm phù hợp nhất dựa trên giá cả và mô tả.
    4. Trả lời các câu hỏi của người dùng về sản phẩm một cách chi tiết và hữu ích.
    Hãy sử dụng ngôn ngữ thân thiện và dễ hiểu.""",
    function_map={"search_ivymoda": search_ivymoda}
)

user_proxy = autogen.UserProxyAgent(
    name="Shopper",
    system_message="Bạn là một người mua sắm đang tìm kiếm sản phẩm trên Ivymoda với giá dưới 500k. Hãy đặt câu hỏi và yêu cầu gợi ý từ Ivymoda_Agent.",
    human_input_mode="TERMINATE",
    code_execution_config={"use_docker": False}
)

# Tạo một GroupChat
groupchat = autogen.GroupChat(
    agents=[user_proxy, ivymoda_agent],
    messages=[],
    max_round=10
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list},
    system_message="""Bạn đang quản lý một cuộc trò chuyện về mua sắm trên Ivymoda. Nhiệm vụ của bạn là:
    1. Điều phối cuộc trò chuyện giữa người mua và Ivymoda_Agent.
    2. Đảm bảo Ivymoda_Agent cung cấp thông tin chính xác và hữu ích về sản phẩm.
    3. Tổng hợp các đề xuất sản phẩm và giúp người mua đưa ra quyết định cuối cùng.
    4. Kết thúc cuộc trò chuyện khi người mua đã có đủ thông tin cần thiết."""
)

# Khởi chạy cuộc trò chuyện
user_proxy.initiate_chat(
    manager,
    message="Chào bạn! Tôi đang tìm kiếm một số sản phẩm thời trang nữ trên Ivymoda với giá dưới 500k. Bạn có thể giúp tôi không?"
)

print("Cuộc trò chuyện về mua sắm trên Ivymoda đã kết thúc! 🛍️👚")