# config.py
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv(".env")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
DATA_PRODUCT_PATH = "data/products.db"
DATA_TEXT_PATH = "data/policy.txt"
STORE_DIRECTORY = "data/datastore"

# Embeddings
load_dotenv(".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")