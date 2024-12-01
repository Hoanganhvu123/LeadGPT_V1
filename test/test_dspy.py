
import os
from dotenv import load_dotenv
import dspy
from dspy.teleprompt import Teleprompter
from dspy import GROQ

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Set up the Groq model
groq_model = GROQ(model="llama-3.2-90b-text-preview")
dspy.configure(lm=groq_model)

# Define a simple question-answering module
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate_answer(question=question)

# Define the QA pipeline
class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = SimpleQA()

    def forward(self, question):
        return self.generate(question=question)

# Compile and optimize the QA pipeline
teleprompter = Teleprompter(QA())
optimized_qa = teleprompter.compile(metric=dspy.Accuracy())

# Example usage
question = "What is the capital of France?"
answer = optimized_qa(question)
print(f"Question: {question}")
print(f"Answer: {answer.answer}")

# You can add more questions and process them in a loop if needed

print("All done! 🎉 Hope this helps, buddy!")
