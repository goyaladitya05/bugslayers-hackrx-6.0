from google.generativeai import list_models
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

models = list_models()
for model in models:
    print(model.name)