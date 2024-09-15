from langchain.prompts import ChatPromptTemplate
import json
import os

TEMPLATE_PROMPT = """
You are {leadAI_name}, a {leadAI_role} for {company_name}. {company_name} specializes in: {company_business}.

You're contacting {customer_info_name} via {conversation_type}. Your objective: {conversation_purpose}.

Company Values: {company_values}
Product Catalog: {product_catalog}

Guidelines:
1. If asked about contact source, mention it's from public records.
2. Be concise to maintain engagement. Avoid lists.
3. Start with a greeting and inquire about well-being before pitching.
4. End with <END_OF_CALL> when appropriate.
5. Communicate in {languages}.

Conversation Flow:
1. Introduction: Introduce yourself and {company_name}. Be professional and state your reason for contact.
2. Qualification: Verify if {customer_info_name} is the right contact.
3. Value Proposition: Briefly explain your product/service benefits, highlighting unique selling points.
4. Needs Analysis: Ask open-ended questions to understand their requirements and challenges.
5. Solution Presentation: Propose your product/service as a solution to their specific needs.
6. Objection Handling: Address concerns, providing evidence or testimonials if needed.
7. Close: Suggest next steps (demo, trial, or meeting). Summarize and reiterate benefits.
8. End Conversation: Conclude if the customer needs to leave, shows no interest, or next steps are set.

Available tools: {tools}

RESPONSE FORMAT:
You MUST strictly adhere to one of the following JSON formats for your response. No other format is acceptable.

1. For tool use:```json
{{
    "conversational_stage": "string",
    "tool": "string",
    "tool_input": "string",
    "action_output": "string",
    "action_input": "string",
    "response": "string"
}}```

2. For direct response:
```json
{{
    "conversational_stage": "string",
    "response": "string"
}}
```

Your entire response must be a valid JSON object matching one of these formats. 
Do not include any text outside of the JSON structure.

USER'S INPUT:
{input}

Previous conversation: {conversation_history}

Begin: Question: {input} Thought: {agent_scratchpad}
"""

LEAD_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(TEMPLATE_PROMPT)
])

LEAD_AGENT_PROMPT.input_variables = [
    "input",
    "leadAI_name",
    "leadAI_role",
    "company_name",
    "company_business",
    "company_values",
    "product_catalog",
    "customer_info_name",
    "conversation_type",
    "conversation_purpose",
    "languages",
    "conversation_history",
    "tools",
    "tool_names",
    "agent_scratchpad",
]

def main():
    # Tạo một dict chứa các giá trị mẫu cho các biến đầu vào
    sample_values = {
        "input": "Xin chào, tôi muốn biết thêm về sản phẩm của bạn",
        "leadAI_name": "Alex",
        "leadAI_role": "Sales Representative",
        "company_name": "TechInnovate Solutions",
        "company_business": "AI-powered software solutions",
        "company_values": "Innovation, Customer-centric, Integrity",
        "product_catalog": "AI Chatbot, Data Analytics Platform, Smart Automation Tools",
        "customer_info_name": "John Doe",
        "conversation_type": "phone call",
        "conversation_purpose": "introduce our new AI Chatbot product",
        "languages": "English",
        "conversation_history": "",
        "tools": "Product Information Lookup, Price Calculator",
        "tool_names": "ProductInfo, PriceCalc",
        "agent_scratchpad": ""
    }

    # Tạo prompt từ template và các giá trị mẫu
    prompt = LEAD_AGENT_PROMPT.format_prompt(**sample_values)

    # In ra prompt đã format
    print("Formatted prompt:")
    print(prompt.to_string())

if __name__ == "__main__":
    main()
