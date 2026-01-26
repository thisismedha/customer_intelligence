"""
dashboard.py
Streamlit dashboard with conversational interface and insights display
FIXED: Tab switching, conversation titles, state persistence, and quick start buttons
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

from conversational_agent import ConversationalEmailAgent
from conversation_manager import ConversationManager


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Email Intelligence Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .chat-message-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .chat-message-ai {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "agent" not in st.session_state:
    st.session_state.agent = ConversationalEmailAgent()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "messages_cache" not in st.session_state:
    st.session_state.messages_cache = {}

if "pending_user_message" not in st.session_state:
    st.session_state.pending_user_message = None

# Persist state across reloads using query params
if "initialized" not in st.session_state:
    query_params = st.query_params
    if "session_id" in query_params:
        st.session_state.current_session_id = query_params["session_id"]
    st.session_state.initialized = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_conversation_messages(session_id: str):
    """Load messages from cache or database"""
    if session_id not in st.session_state.messages_cache:
        st.session_state.messages_cache[session_id] = st.session_state.agent.get_conversation_history(session_id)
    return st.session_state.messages_cache[session_id]


def invalidate_messages_cache(session_id: str):
    """Clear cached messages for a session"""
    if session_id in st.session_state.messages_cache:
        del st.session_state.messages_cache[session_id]


def update_url_params():
    """Update URL parameters to persist state"""
    if st.session_state.current_session_id:
        st.query_params.update({
            "session_id": st.session_state.current_session_id
        })
    else:
        st.query_params.clear()


def switch_to_conversation(session_id: str):
    """Switch to a conversation"""
    st.session_state.current_session_id = session_id
    update_url_params()


# ============================================================================
# SIDEBAR: CONVERSATION MANAGEMENT
# ============================================================================

def render_sidebar():
    """Render the sidebar with conversation list and controls"""
    
    with st.sidebar:
        st.markdown("### 💬 Conversations")
        
        # New conversation button - DON'T create session here, just clear current
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("➕ New Conversation", use_container_width=True, key="new_conv_btn"):
                # Clear current session to show the quick start interface
                st.session_state.current_session_id = None
                st.session_state.pending_user_message = None
                update_url_params()
                st.rerun()
        
        with col2:
            if st.button("🔄", help="Refresh list", key="refresh_btn"):
                # Clear cache and reload
                st.session_state.messages_cache = {}
                st.rerun()
        
        st.markdown("---")
        
        # List conversations
        conversations = st.session_state.agent.list_conversations(limit=50)
        
        if not conversations:
            st.info("No conversations yet. Start a new one!")
        else:
            for idx, conv in enumerate(conversations):
                is_active = conv["session_id"] == st.session_state.current_session_id
                
                # Conversation item container
                with st.container():
                    # Title button (full width)
                    if st.button(
                        f"{'📍' if is_active else '💬'} {conv['title'][:40]}",
                        key=f"conv_{conv['session_id']}_{idx}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        switch_to_conversation(conv["session_id"])
                        st.rerun()
                    
                    # Metadata and action buttons in single row
                    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
                    
                    with col1:
                        st.caption(f"{conv['message_count']} msgs • {conv['last_updated'][:10]}")
                    
                    with col2:
                        # Rename button
                        if st.button("✏️", key=f"rename_btn_{conv['session_id']}_{idx}", help="Rename"):
                            st.session_state.rename_session_id = conv['session_id']
                            st.session_state.rename_current_title = conv['title']
                    
                    with col3:
                        # Delete button
                        if st.button("🗑️", key=f"delete_btn_{conv['session_id']}_{idx}", help="Delete"):
                            st.session_state.delete_confirm_id = conv['session_id']
                    
                    with col4:
                        # Spacer for alignment
                        st.write("")
                
                st.markdown("<div style='margin-bottom: 0.3rem'></div>", unsafe_allow_html=True)
        
        # Delete confirmation dialog
        if hasattr(st.session_state, "delete_confirm_id") and st.session_state.delete_confirm_id:
            st.markdown("---")
            st.warning("⚠️ Delete this conversation?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete", type="primary", use_container_width=True, key="confirm_delete"):
                    st.session_state.agent.delete_conversation(st.session_state.delete_confirm_id)
                    invalidate_messages_cache(st.session_state.delete_confirm_id)
                    if st.session_state.current_session_id == st.session_state.delete_confirm_id:
                        st.session_state.current_session_id = None
                        update_url_params()
                    st.session_state.delete_confirm_id = None
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True, key="cancel_delete"):
                    st.session_state.delete_confirm_id = None
                    st.rerun()
        
        # Rename dialog
        if hasattr(st.session_state, "rename_session_id") and st.session_state.rename_session_id:
            st.markdown("---")
            st.markdown("**Rename Conversation**")
            new_title = st.text_input(
                "New title:", 
                value=st.session_state.get("rename_current_title", ""),
                key="new_title_input"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save", type="primary", use_container_width=True, key="save_rename"):
                    if new_title and new_title != st.session_state.get("rename_current_title"):
                        st.session_state.agent.rename_conversation(
                            st.session_state.rename_session_id,
                            new_title
                        )
                    st.session_state.rename_session_id = None
                    if hasattr(st.session_state, "rename_current_title"):
                        del st.session_state.rename_current_title
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True, key="cancel_rename"):
                    st.session_state.rename_session_id = None
                    if hasattr(st.session_state, "rename_current_title"):
                        del st.session_state.rename_current_title
                    st.rerun()


# ============================================================================
# MAIN AREA: TABS
# ============================================================================

def render_overview_tab():
    """Render the overview/insights tab"""
    
    st.markdown("<div class='main-header'>📊 Email Intelligence Overview</div>", unsafe_allow_html=True)
    
    # Load database stats
    conn = sqlite3.connect("email_intel.db")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_emails = pd.read_sql("SELECT COUNT(*) as count FROM emails", conn).iloc[0]['count']
        st.metric("📧 Total Emails", f"{total_emails:,}")
    
    with col2:
        total_brands = pd.read_sql("SELECT COUNT(DISTINCT brand) as count FROM emails", conn).iloc[0]['count']
        st.metric("🏢 Brands", f"{total_brands}")
    
    with col3:
        avg_discount = pd.read_sql(
            "SELECT AVG(discount_percent) as avg FROM emails WHERE discount_percent IS NOT NULL",
            conn
        ).iloc[0]['avg']
        st.metric("💰 Avg Discount", f"{avg_discount:.1f}%")
    
    with col4:
        total_insights = pd.read_sql("SELECT COUNT(*) as count FROM generated_insights", conn).iloc[0]['count']
        st.metric("💡 Insights", f"{total_insights}")
    
    st.markdown("---")
    
    # Key Insights
    st.markdown("### 🔍 Key Findings")
    
    insights = pd.read_sql("""
        SELECT category, finding, metric_value, confidence, viz_path, created_at
        FROM generated_insights
        ORDER BY created_at DESC
        LIMIT 10
    """, conn)
    
    if not insights.empty:
        for _, insight in insights.iterrows():
            with st.expander(f"**{insight['category'].replace('_', ' ').title()}** - {insight['finding'][:60]}...", expanded=False):
                st.markdown(f"**Finding:** {insight['finding']}")
                
                if pd.notna(insight['metric_value']):
                    st.metric("Metric", f"{insight['metric_value']:.2f}")
                
                st.caption(f"Confidence: {insight['confidence']} • Created: {insight['created_at'][:10]}")
                
                # Show visualization if exists
                if pd.notna(insight['viz_path']) and Path(insight['viz_path']).exists():
                    st.components.v1.html(
                        open(insight['viz_path'], 'r').read(),
                        height=500,
                        scrolling=True
                    )
    else:
        st.info("No insights generated yet. Run the agent or ask questions in the chat!")
    
    st.markdown("---")
    
    # Quick Charts
    st.markdown("### 📈 Quick Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average discount by brand
        brand_discounts = pd.read_sql("""
            SELECT brand, AVG(discount_percent) as avg_discount
            FROM emails
            WHERE discount_percent IS NOT NULL
            GROUP BY brand
            ORDER BY avg_discount DESC
            LIMIT 10
        """, conn)
        
        if not brand_discounts.empty:
            fig = px.bar(
                brand_discounts,
                x='brand',
                y='avg_discount',
                title="Top 10 Brands by Average Discount",
                labels={'avg_discount': 'Avg Discount %', 'brand': 'Brand'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Urgency distribution
        urgency_dist = pd.read_sql("""
            SELECT urgency_level, COUNT(*) as count
            FROM emails
            GROUP BY urgency_level
        """, conn)
        
        if not urgency_dist.empty:
            fig = px.pie(
                urgency_dist,
                values='count',
                names='urgency_level',
                title="Urgency Level Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    conn.close()


def render_chat_tab():
    """Render the conversational chat interface"""
    
    st.markdown("<div class='main-header'>💬 Chat with Your Data</div>", unsafe_allow_html=True)
    
    # Check if we have a current session
    if not st.session_state.current_session_id:
        st.info("👈 Select a conversation from the sidebar or create a new one to start chatting!")
        
        # Quick start buttons
        st.markdown("### Quick Start Questions")
        col1, col2, col3 = st.columns(3)
        
        quick_questions = [
            "Which brand discounts most aggressively?",
            "Show me urgency patterns by brand",
            "Are discounts increasing over time?",
            "Which brands use reactivation tactics?",
            "Compare discount trends across brands",
            "Find brands with fake urgency"
        ]
        
        for i, question in enumerate(quick_questions):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    # Start new conversation with this question
                    with st.spinner("🤔 Processing..."):
                        # Use chat() which creates new session AND generates title
                        result = st.session_state.agent.chat(question)
                        session_id = result['session_id']
                        
                        # Cache the initial exchange
                        st.session_state.messages_cache[session_id] = [
                            {'role': 'human', 'content': question},
                            {'role': 'ai', 'content': result['response'], 'visualizations': result.get('visualizations')}
                        ]
                        
                        # Switch to the new conversation
                        switch_to_conversation(session_id)
                        st.rerun()
        
        return
    
    # Load conversation history from cache
    history = load_conversation_messages(st.session_state.current_session_id)
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        for msg in history:
            if msg['role'] == 'human':
                st.markdown(f"""
                <div class='chat-message-user'>
                    <strong>👤 You</strong><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            
            elif msg['role'] == 'ai':
                st.markdown(f"""
                <div class='chat-message-ai'>
                    <strong>🤖 Assistant</strong><br>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show visualizations if any
                if msg.get('visualizations'):
                    import json
                    viz_paths = json.loads(msg['visualizations']) if isinstance(msg['visualizations'], str) else msg['visualizations']
                    for viz_path in viz_paths:
                        if Path(viz_path).exists():
                            st.components.v1.html(
                                open(viz_path, 'r').read(),
                                height=500,
                                scrolling=True
                            )
        
        # Show pending user message immediately (optimistic UI)
        if st.session_state.pending_user_message:
            st.markdown(f"""
            <div class='chat-message-user'>
                <strong>👤 You</strong><br>
                {st.session_state.pending_user_message}
            </div>
            """, unsafe_allow_html=True)
            
            # Show thinking indicator
            with st.spinner("🤔 Thinking..."):
                # Process the message
                result = st.session_state.agent.continue_conversation(
                    st.session_state.current_session_id,
                    st.session_state.pending_user_message
                )
                
                # Update cache with both user message and AI response
                st.session_state.messages_cache[st.session_state.current_session_id].extend([
                    {'role': 'human', 'content': st.session_state.pending_user_message},
                    {'role': 'ai', 'content': result['response'], 'visualizations': result.get('visualizations')}
                ])
                
                # Clear pending message
                st.session_state.pending_user_message = None
                st.rerun()
    
    # Chat input
    st.markdown("---")
    
    user_input = st.chat_input("Ask a question about your email data...")
    
    if user_input:
        # Set pending message for optimistic UI update
        st.session_state.pending_user_message = user_input
        st.rerun()
    
    # Conversation stats in sidebar
    with st.sidebar:
        if st.session_state.current_session_id:
            st.markdown("---")
            st.markdown("### 📊 Current Conversation")
            stats = st.session_state.agent.get_conversation_stats(st.session_state.current_session_id)
            st.metric("Messages", stats['total_messages'])
            st.metric("Visualizations", stats['visualizations_created'])
            
            if st.button("📥 Export Conversation", use_container_width=True, key="export_conv"):
                markdown = st.session_state.agent.export_conversation(
                    st.session_state.current_session_id,
                    format="markdown"
                )
                st.download_button(
                    "Download Markdown",
                    markdown,
                    file_name=f"conversation_{st.session_state.current_session_id[:8]}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_md"
                )


def render_data_explorer_tab():
    """Render the data explorer tab"""
    
    st.markdown("<div class='main-header'>🔍 Data Explorer</div>", unsafe_allow_html=True)
    
    conn = sqlite3.connect("email_intel.db")
    
    # Table selector
    table = st.selectbox("Select Table", ["emails", "brand_insights", "generated_insights"])
    
    # Filters
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if table == "emails":
            brands = pd.read_sql("SELECT DISTINCT brand FROM emails ORDER BY brand", conn)
            selected_brands = st.multiselect("Brands", brands['brand'].tolist())
    
    with col2:
        if table == "emails":
            discount_types = pd.read_sql("SELECT DISTINCT discount_type FROM emails", conn)
            selected_types = st.multiselect("Discount Type", discount_types['discount_type'].tolist())
    
    with col3:
        limit = st.number_input("Rows to display", min_value=10, max_value=1000, value=100)
    
    # Build query
    query = f"SELECT * FROM {table}"
    where_clauses = []
    
    if table == "emails" and selected_brands:
        where_clauses.append(f"brand IN ({','.join(['?' for _ in selected_brands])})")
    
    if table == "emails" and selected_types:
        where_clauses.append(f"discount_type IN ({','.join(['?' for _ in selected_types])})")
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += f" LIMIT {limit}"
    
    # Execute query
    params = []
    if selected_brands:
        params.extend(selected_brands)
    if selected_types:
        params.extend(selected_types)
    
    df = pd.read_sql(query, conn, params=params if params else None)
    
    # Display
    st.markdown(f"### {table.title()} Table")
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        file_name=f"{table}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_csv"
    )
    
    conn.close()


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Render sidebar
    render_sidebar()
    
    # Create tabs - Streamlit will remember which tab is active
    tab_overview, tab_chat, tab_explorer = st.tabs(["📊 Overview", "💬 Chat", "🔍 Data Explorer"])
    
    # Render all tabs
    with tab_overview:
        render_overview_tab()
    
    with tab_chat:
        render_chat_tab()
    
    with tab_explorer:
        render_data_explorer_tab()


if __name__ == "__main__":
    main()