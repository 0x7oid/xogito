from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

CALL_TIMEOUT_SECONDS = 300 

DEFAULT_MODEL = "gemini-2.5-flash"
load_dotenv()
API_KEY = os.getenv("API_KEY")

client = genai.Client(api_key = API_KEY)

def ask_llm(prompt,schema, model=DEFAULT_MODEL):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            timeout= CALL_TIMEOUT_SECONDS
        )
    )
    return response.text
