
---
title: Email Intelligence Dashboard
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.29.0
app_file: dashboard.py
pinned: false
---

# 📧 Email Intelligence Dashboard

An AI-powered analytics dashboard for promotional email data. Analyze discount trends, urgency tactics, brand strategies, and more.

## 🎯 Features

- **💬 Interactive Chat**: Ask questions about your email data in natural language
- **📊 Visual Analytics**: Automated insights with interactive Plotly charts
- **🔍 Data Explorer**: Browse and filter your email database
- **📈 Brand Insights**: Compare strategies across brands
- **⚡ Real-time Analysis**: LLM-powered conversational queries

## 🛠️ Built With

- **Streamlit** - Interactive web interface
- **LangChain** - AI agent orchestration
- **Google Gemini** - Large language model
- **SQLite** - Local database
- **Plotly** - Data visualizations

## 💡 Example Questions

- "Which brands use the most urgency?"
- "Show me discount trends over time"
- "Compare average discounts by brand"
- "Which brands should I unsubscribe from?"

## 🚀 Usage

1. Navigate to the **Chat** tab
2. Type your question or click a quick start button
3. Get AI-powered insights and visualizations
4. Explore the **Overview** and **Data Explorer** tabs for more details

---

*Note: This demo uses a pre-populated sample database. To analyze your own emails, you'll need to run the email extraction pipeline locally.*

# Project Setup

This project uses Python and a virtual environment to manage dependencies.

---

## Setup Instructions
### 1️⃣ Clone the repository

git clone git@github.com:thisismedha/customer_intelligence.git
cd customer_intelligence

### 2️⃣ Create a virtual environment
python3 -m venv venv

### 3️⃣ Activate the virtual environment

#### macOS / Linux

source venv/bin/activate


#### Windows (PowerShell)

venv\Scripts\Activate.ps1


You should see (venv) in your terminal.

### 4️⃣ Install dependencies
pip install -r requirements.txt

#### Deactivate the environment
deactivate


