"""
Streamlit Frontend for Email Intelligence Analyst
Run with: streamlit run app.py
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add backend directory to path to import the analyst
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from intelligence_analyst import SQLIntelligenceAnalyst

# Page config
st.set_page_config(
    page_title="Email Intelligence Analyst",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyst' not in st.session_state:
    with st.spinner('🤖 Initializing Email Intelligence Analyst...'):
        # Path to the JSON file in backend folder
        intelligence_file = Path(__file__).parent.parent / "backend" / "email_intelligence.json"
        
        if not intelligence_file.exists():
            st.error(f"❌ Could not find email_intelligence.json at: {intelligence_file}")
            st.stop()
        
        st.session_state.analyst = SQLIntelligenceAnalyst(str(intelligence_file))
    st.success('✓ Analyst ready!')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'show_sql' not in st.session_state:
    st.session_state.show_sql = False

# Header
st.title("📧 Email Intelligence Analyst")
st.markdown("Ask questions about your promotional emails in natural language")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Show SQL toggle
    st.session_state.show_sql = st.checkbox("Show SQL queries", value=st.session_state.show_sql)
    
    st.divider()
    
    # Stats
    st.header("📊 Database Stats")
    analyst = st.session_state.analyst
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Emails", analyst._count_emails())
    with col2:
        st.metric("Brands", analyst._count_brands())
    
    st.divider()
    
    # Example questions
    st.header("💡 Example Questions")
    example_questions = [
        "How many emails from Target?",
        "What's the average discount?",
        "Which brand emails most?",
        "Show weekend emails",
        "Brands with fake urgency?",
        "Highest discounts?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"example_{i}"):
            st.session_state.user_input = question
            st.rerun()
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # MVP Questions
    st.divider()
    if st.button("📋 Run MVP Analysis", use_container_width=True):
        with st.spinner('Running comprehensive analysis...'):
            results = analyst.answer_mvp_questions()
            st.session_state.show_mvp = True
            st.session_state.mvp_results = results
        st.success('✓ Analysis complete!')
        st.rerun()

# Main chat area
st.header("💬 Chat")

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message['content']}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong> {message['content']}
            </div>
        """, unsafe_allow_html=True)
        
        # Show SQL if enabled
        if st.session_state.show_sql and 'sql' in message:
            with st.expander("📊 SQL Query"):
                st.code(message['sql'], language='sql')

# Chat input
user_input = st.chat_input("Ask a question about your emails...")

# Handle example question from sidebar
if 'user_input' in st.session_state and st.session_state.user_input:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    # Get response
    with st.spinner('🔍 Analyzing...'):
        try:
            # Generate SQL
            sql = analyst.query_to_sql(user_input)
            sql = analyst._validate_sql_quotes(sql)
            
            # Execute query
            results = analyst.execute_query(sql)
            
            if results.empty:
                response = "No results found for this query."
            else:
                # Get LLM interpretation
                from langchain_core.messages import HumanMessage, SystemMessage
                
                results_str = results.to_string(index=False, max_rows=20)
                
                interpretation_prompt = f"""Answer the user's question based on these query results.

USER QUESTION: {user_input}

SQL QUERY USED: {sql}

QUERY RESULTS:
{results_str}

Provide a clear, natural language answer. Include specific numbers and insights.
If the data shows trends or patterns, mention them."""

                messages = [
                    SystemMessage(content="You are a helpful analyst. Provide clear, accurate answers based on data."),
                    HumanMessage(content=interpretation_prompt)
                ]
                
                llm_response = analyst.llm.invoke(messages)
                response = llm_response.content
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'sql': sql
            })
            
        except Exception as e:
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': f"❌ Error: {str(e)}\nTry rephrasing your question."
            })
    
    st.rerun()

# MVP Results section
if 'show_mvp' in st.session_state and st.session_state.show_mvp:
    st.divider()
    st.header("📋 MVP Analysis Results")
    
    results = st.session_state.mvp_results
    
    # Question 1
    with st.expander("📊 Q1: Discount Aggression Over Time", expanded=True):
        if results['question_1']:
            import pandas as pd
            df = pd.DataFrame(results['question_1'])
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            if not df.empty:
                st.bar_chart(df.set_index('brand_name')['change'])
        else:
            st.info("No data available")
    
    # Question 3
    with st.expander("📅 Q3: Cyclical Clustering", expanded=True):
        if results['question_3']:
            df = pd.DataFrame(results['question_3'])
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            if not df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Month-End %")
                    st.bar_chart(df.set_index('brand_name')['month_end_pct'])
                with col2:
                    st.subheader("Quarter-End %")
                    st.bar_chart(df.set_index('brand_name')['quarter_end_pct'])
        else:
            st.info("No data available")
    
    # Question 4
    with st.expander("⚡ Q4: Urgency Authenticity", expanded=True):
        if results['question_4']:
            df = pd.DataFrame(results['question_4'])
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            if not df.empty:
                st.bar_chart(df.set_index('brand_name')['avg_urgency'])
        else:
            st.info("No data available")
    
    if st.button("Close MVP Results"):
        st.session_state.show_mvp = False
        st.rerun()

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Email Intelligence Analyst MVP | Powered by SQLite + Gemini</small>
    </div>
""", unsafe_allow_html=True)