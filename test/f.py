from typing import List, Dict, Optional
from langchain.graphs import Graph
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

# 1. Define States và Models
class ConversationState(BaseModel):
    current_stage: str = "1"
    customer_name: Optional[str] = None
    needs: Dict = {}
    buying_signals: List[str] = []
    conversation_history: List[Dict] = []

# 2. Define Nodes
class BaseNode:
    def __init__(self, llm):
        self.llm = llm
        
class GreetingNode(BaseNode):
    def run(self, state: ConversationState) -> ConversationState:
        prompt = ChatPromptTemplate.from_template("""
        Bạn là AI assistant chuyên nghiệp. Hãy chào khách hàng theo guidelines:
        - Thể hiện sự nhiệt tình, sẵn sàng hỗ trợ
        - Giọng điệu thân thiện, gần gũi
        - Thể hiện chuyên môn về sản phẩm
        
        Chat history: {history}
        Customer input: {input}
        
        Response:
        """)
        # Process và update state
        return state

class AssessmentNode(BaseNode):
    def run(self, state: ConversationState) -> ConversationState:
        prompt = ChatPromptTemplate.from_template("""
        Đánh giá khách hàng dựa trên:
        - Pain points và nhu cầu thực sự
        - Mức độ hiểu biết về sản phẩm
        - Ngân sách và thời gian
        - Các yếu tố quyết định
        
        Dựa vào đánh giá, xác định khách thuộc nhóm:
        1. POTENTIAL - Sẵn sàng mua
        2. NURTURING - Cần thời gian
        
        Chat history: {history}
        Current state: {state}
        Customer input: {input}
        
        Response:
        """)
        return state

# Tương tự cho các node khác: P3, P4, P5, N3, N4, N5

# 3. Define Edges và Transitions
class EdgeEvaluator:
    @staticmethod
    def evaluate(source: str, target: str, state: ConversationState) -> bool:
        # Logic chuyển stage dựa trên:
        # - Buying signals trong input
        # - Stage hiện tại
        # - Trạng thái conversation
        if source == "2":
            if "buying_signals" in state.needs:
                return target.startswith("P")
            return target.startswith("N")
            
        # Logic chuyển từ N sang P khi phát hiện buying signals
        if source.startswith("N") and target.startswith("P"):
            return len(state.buying_signals) > 0
            
        return True

# 4. Build Graph
class LeadGraph:
    def __init__(self):
        self.graph = Graph()
        self.setup_nodes()
        self.setup_edges()
        
    def setup_nodes(self):
        # Add các node tương ứng với các stage
        self.graph.add_node("1", GreetingNode)
        self.graph.add_node("2", AssessmentNode)
        # Add các node P3-P5, N3-N5
        
    def setup_edges(self):
        # Add edges cho phép chuyển stage
        self.graph.add_edge("1", "2")
        self.graph.add_edge("2", "P3")
        self.graph.add_edge("2", "N3")
        # Add edges giữa các stage và cross-track edges
        
    def process(self, input: str, state: ConversationState) -> ConversationState:
        current_node = self.graph.get_node(state.current_stage)
        new_state = current_node.run(state)
        
        # Evaluate transitions
        next_stage = self.evaluate_next_stage(new_state)
        new_state.current_stage = next_stage
        
        return new_state

# 5. Main Processing Loop
class LeadBot:
    def __init__(self):
        self.graph = LeadGraph()
        self.state = ConversationState()
        
    async def process_message(self, message: str):
        # Update state với input mới
        self.state.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Process qua graph
        new_state = self.graph.process(message, self.state)
        self.state = new_state
        
        # Generate response
        response = await self.generate_response()
        
        return response