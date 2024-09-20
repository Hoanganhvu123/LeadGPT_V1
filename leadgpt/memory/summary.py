from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel

from leadgpt.memory.prompt import LEAD_PROMPT_TEMPLATE

class LeadSummaryMemory(BaseModel):
    """Simplified Lead Summary Memory"""

    llm: BaseLanguageModel
    prompt: BasePromptTemplate = LEAD_PROMPT_TEMPLATE
    buffer: str = ""

    def predict_new_summary(self, input: dict) -> str:
        """Predict new summary based on existing summary and new lines"""
        chain = self.prompt | self.llm
        result = chain.invoke(input)
        return result.content if hasattr(result, 'content') else str(result)

    def update_summary(self, new_lines: str) -> None:
        """Update the summary with new lines"""
        self.buffer = self.predict_new_summary({
            "customer_info": self.buffer,
            "new_lines": new_lines
        })

    def get_summary(self) -> str:
        """Get the current summary"""
        return self.buffer

    def clear(self) -> None:
        """Clear the summary"""
        self.buffer = ""