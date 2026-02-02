# llm.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st

# Get API key from Streamlit secrets
API_KEY = st.secrets.get("GOOGLE_API_KEY")

def get_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0
):
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=API_KEY,
        temperature=temperature,
    )
