from langchain_core.prompts.prompt import PromptTemplate

TEMPLATE = """
You are an AI assistant tasked with updating a conversation summary. 
Update the summary to include new conversation lines while retaining relevant existing details. 
Use these specifics:

- Previous summary: customer_info = [{customer_info}]
- New conversation lines: new_lines = [{new_lines}]

Update the summary to include only these fields: Customer Name, Email, Phone, and Products of Interest. 
Include information only if explicitly mentioned in the conversation.
If a field is not mentioned, mark it "None". 
Do not add explanations, extra context, or additional information. Do not create information if new_lines is empty.

### FORMAT OUTPUT ### :

Customer Name: None
Email: None
Phone: None
Products of Interest: None

## IMPORTANT ### : 

Update based only on "HUMAN MESSAGE".
"""

LEAD_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["customer_info", "new_lines"],
    template=TEMPLATE
)
