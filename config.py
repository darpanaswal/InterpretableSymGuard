import os
from dotenv import load_dotenv

load_dotenv()

openai_token = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_API_KEY")

if not openai_token or not hf_token:
    raise ValueError("API keys are not set in environment variables")