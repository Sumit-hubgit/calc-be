from dotenv import load_dotenv
import os
load_dotenv()

SERVER_URL = '0.0.0.0'  # Bind to all interfaces
PORT = os.getenv("PORT", "8000")  # Use the PORT environment variable or default to 8000

ENV = 'dev'

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
