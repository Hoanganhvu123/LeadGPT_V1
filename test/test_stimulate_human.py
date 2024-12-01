import os
from dotenv import load_dotenv
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy import GROQ
from dspy.primitives import Example
from langchain_groq import ChatGroq
from leadgpt.agent.lead_agent import LeadGPT
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")

# Set up the Groq model
groq_model = GROQ(model="llama-3.2-90b-text-preview", api_key=os.environ['GROQ_API_KEY'])
dspy.configure(lm=groq_model)

# Define state
class State(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The messages in the conversation"]
    next: Annotated[str, "Who goes next, 'bot1' or 'bot2'"]
    bot1_name: Annotated[str, "Name of the first bot"]
    bot2_name: Annotated[str, "Name of the second bot"]

# Define functions for the graph
def bot1(state: State) -> State:
    bot1_name = state["bot1_name"]
    llm_groq = ChatGroq(temperature=0.3, model="llama-3.2-90b-text-preview")
    lead = LeadGPT(
        llm=llm_groq,
        verbose=False,
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
    
    bot2_message = state["messages"][-1].content.split(": ", 1)[1]
    lead.human_step(bot2_message)
    lead.determine_conversation_stage()
    lead.update_customer_info()
    bot1_response = lead.agent_step()
    
    state["messages"].append(AIMessage(content=f"{bot1_name}: {bot1_response}"))
    state["next"] = "bot2"
    return state

def bot2(state: State) -> State:
    bot2_name = state["bot2_name"]
    llm_groq = ChatGroq(temperature=0.3, model="llama-3.2-90b-text-preview")
    
    if state["messages"]:
        bot1_message = state["messages"][-1].content.split(": ", 1)[1]
        
        bot2_response = llm_groq.invoke(
            f"""Bạn là một khách hàng đang tìm kiếm quần áo tại DaisyShop. 
            Hãy trả lời tin nhắn của nhân viên bán hàng một cách tự nhiên và đặt câu hỏi về sản phẩm nếu cần.
            Hãy nhớ và sử dụng thông tin từ các lượt tương tác trước đó.
            Duy trì tính nhất quán trong cuộc trò chuyện.
            
            Tin nhắn của nhân viên bán hàng: {bot1_message}
            Trả lời:"""
        )
    else:
        bot2_response = llm_groq.invoke(
            """Bạn là một khách hàng đang tìm kiếm quần áo tại DaisyShop. 
            Hãy bắt đầu cuộc hội thoại bằng cách chào nhân viên bán hàng và hỏi về một sản phẩm cụ thể.
            Trả lời:"""
        )
    
    state["messages"].append(AIMessage(content=f"{bot2_name}: {bot2_response.content}"))
    state["next"] = "bot1"
    return state

def should_end(state: State) -> bool:
    return len(state["messages"]) >= 10 or "Tạm biệt" in state["messages"][-1].content

# Create the graph
workflow = StateGraph(State)

# Define the nodes
workflow.add_node("bot1", bot1)
workflow.add_node("bot2", bot2)

# Define the edges
workflow.add_edge("bot1", "bot2")
workflow.add_edge("bot2", "bot1")

# Set the entry point
workflow.set_entry_point("bot2")

# Add conditional edges
workflow.add_conditional_edges(
    "bot1",
    should_end,
    {
        True: END,
        False: "bot2",
    },
)

workflow.add_conditional_edges(
    "bot2",
    should_end,
    {
        True: END,
        False: "bot1",
    },
)

# Compile the graph
app = workflow.compile()

class ConversationLabeler(dspy.Signature):
    """Label the conversation and suggest improvements."""
    conversation = dspy.InputField()
    label = dspy.OutputField(desc="Label indicating the quality of the conversation")
    suggestions = dspy.OutputField(desc="Suggestions for improving the LeadGPT prompt")

class OptimizeLeadGPTPrompt(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought("conversation -> label")
        self.suggest = dspy.ChainOfThought("conversation, label -> suggestions")

    def forward(self, conversation):
        label = self.classify(conversation=conversation)
        suggestions = self.suggest(conversation=conversation, label=label.label)
        return label.label, suggestions.suggestions

# Define example conversations and their labels
trainset = [
    Example(
        conversation="""
        DaisyBot: Xin chào! Chào mừng bạn đến với DaisyShop. Tôi có thể giúp gì cho bạn hôm nay?
        Khách hàng: Chào bạn, tôi đang tìm một chiếc áo sơ mi nam.
        DaisyBot: Rất vui khi được hỗ trợ bạn tìm áo sơ mi nam. Chúng tôi có nhiều mẫu áo sơ mi nam đa dạng. Bạn có ưu tiên về màu sắc, kiểu dáng hoặc chất liệu cụ thể nào không?
        Khách hàng: Tôi thích màu xanh nhạt và chất liệu thoáng mát.
        DaisyBot: Tuyệt vời! Chúng tôi có một số lựa chọn phù hợp với yêu cầu của bạn. Đối với áo sơ mi nam màu xanh nhạt và chất liệu thoáng mát, tôi đề xuất:
        1. Áo sơ mi cotton pha linen màu xanh pastel: Rất nhẹ và thoáng khí, phù hợp cho thời tiết nóng.
        2. Áo sơ mi lụa tencel màu xanh nhạt: Mềm mại, thoáng mát và có độ bóng nhẹ.
        Bạn thích kiểu nào hơn?
        Khách hàng: Tôi nghĩ áo cotton pha linen nghe có vẻ phù hợp. Bạn có thể cho tôi biết thêm về giá cả và size không?
        DaisyBot: Tất nhiên! Tôi rất vui khi bạn quan tâm đến áo sơ mi cotton pha linen của chúng tôi. Đây là thông tin chi tiết:
        - Giá: 599.000 VNĐ
        - Size: Có sẵn từ S đến XXL
        - Màu sắc: Xanh pastel như bạn yêu cầu, và còn có các màu khác như trắng, be, xám nhạt
        - Đặc điểm: 70% cotton, 30% linen, cổ đứng, cộc tay
        Bạn muốn thử size nào? Tôi có thể giúp bạn chọn size phù hợp nếu bạn cho biết chiều cao và cân nặng.
        Khách hàng: Tôi cao 1m75 và nặng 70kg. Bạn nghĩ size nào sẽ phù hợp?
        DaisyBot: Với chiều cao 1m75 và cân nặng 70kg của bạn, tôi nghĩ size L sẽ là lựa chọn phù hợp nhất. Size L thường phù hợp với người có chiều cao từ 1m70 đến 1m80 và cân nặng từ 65kg đến 75kg. Tuy nhiên, nếu bạn thích mặc hơi rộng một chút, bạn có thể cân nhắc size XL.
        Bạn có muốn đặt hàng size L không, hay bạn cần thêm thông tin gì nữa?
        Khách hàng: Cảm ơn bạn, tôi sẽ lấy size L. Làm thế nào để đặt hàng?
        DaisyBot: Tuyệt vời! Tôi rất vui khi bạn quyết định chọn áo sơ mi cotton pha linen size L của chúng tôi. Để đặt hàng, bạn có thể thực hiện theo các bước sau:
        1. Truy cập website: www.daisyshop.com
        2. Tìm mục "Áo sơ mi nam" và chọn sản phẩm "Áo sơ mi cotton pha linen màu xanh pastel"
        3. Chọn size L và số lượng bạn muốn mua
        4. Nhấn "Thêm vào giỏ hàng" và tiến hành thanh toán
        5. Điền thông tin giao hàng và chọn phương thức thanh toán
        Nếu bạn gặp bất kỳ khó khăn nào trong quá trình đặt hàng, đừng ngần ngại liên hệ với chúng tôi qua số hotline 1900xxxx. Chúng tôi luôn sẵn sàng hỗ trợ bạn.
        Bạn có cần tôi hướng dẫn thêm về quy trình đặt hàng không?
        Khách hàng: Không, cảm ơn bạn. Tôi nghĩ tôi có thể tự làm được. Cảm ơn sự giúp đỡ của bạn!
        DaisyBot: Rất vui được hỗ trợ bạn! Cảm ơn bạn đã chọn mua sắm tại DaisyShop. Chúc bạn có trải nghiệm mua sắm tuyệt vời. Nếu bạn cần bất kỳ sự hỗ trợ nào trong tương lai, đừng ngần ngại liên hệ với chúng tôi. Chúc bạn một ngày tốt lành!
        Khách hàng: Cảm ơn bạn. Chào tạm biệt!
        """,
        label="Excellent",
        suggestions="The conversation flow is smooth and professional. The LeadGPT prompt is working well, but could be improved by adding more personalized recommendations based on customer preferences and previous purchases if available."
    ).with_inputs("conversation"),
    # Add more examples here
]

def conversation_quality_metric(example, prediction, trace):
    return example.label.lower() == prediction.label.lower()

# Compile and optimize the LeadGPT prompt optimizer
teleprompter = BootstrapFewShot(metric=conversation_quality_metric)
optimized_prompt_optimizer = teleprompter.compile(OptimizeLeadGPTPrompt(), trainset=trainset)

# Main function
def main():
    # Run the simulation
    final_output = None
    state = {
        "messages": [], 
        "next": "bot2",
        "bot1_name": "DaisyBot",
        "bot2_name": "Khách hàng"
    }
    
    while True:
        output = app.invoke(state)
        if "messages" in output:
            message = output["messages"][-1].content
            print(message)
        
        state = output
        
        if output.get("next") == END:
            final_output = output
            break
    
    # Convert the conversation to a string
    conversation = "\n".join([msg.content for msg in final_output["messages"]])
    
    # Optimize prompt
    label, suggestions = optimized_prompt_optimizer(conversation)
    
    print("\nConversation Label:", label)
    print("\nSuggestions for improving LeadGPT prompt:")
    print(suggestions)
    
    # Update LeadGPT prompt based on suggestions
    # Note: This is a simplified example. In practice, you'd need to implement a more sophisticated way to update the prompt.
    llm_groq = ChatGroq(temperature=0.3, model="llama-3.2-90b-text-preview")
    lead_gpt = LeadGPT(
        llm=llm_groq,
        verbose=False,
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
    
    lead_gpt.company_values += f"\n{suggestions}"
    
    print("\nOptimized LeadGPT Prompt:")
    print(lead_gpt.company_values)

if __name__ == "__main__":
    main()