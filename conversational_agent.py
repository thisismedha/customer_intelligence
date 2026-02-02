"""
conversational_agent.py
Session-aware wrapper around the ReAct agent for multi-turn conversations
"""

import uuid
from typing import Optional, Dict, List


from agent import EmailIntelligenceAgent, build_react_agent
from conversation_manager import ConversationManager, generate_session_title
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage


class ConversationalEmailAgent:
    """
    Wrapper around EmailIntelligenceAgent that supports:
    - Multi-session conversations
    - Persistent conversation history
    - Context retention across messages
    - Visualization tracking
    """
    
    def __init__(self, db_path: str = "email_intel.db"):
        self.db_path = db_path
        self.base_agent = EmailIntelligenceAgent(mode="chat")
        self.conversation_manager = ConversationManager(db_path)
        self.graph = build_react_agent()
    
    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            title: Optional custom title
        
        Returns:
            session_id
        """
        return self.conversation_manager.create_session(title)
    
    def list_conversations(self, limit: int = 50) -> List[Dict]:
        """Get all conversation sessions"""
        sessions = self.conversation_manager.list_sessions(limit)
        return [session.to_dict() for session in sessions]
    
    def delete_conversation(self, session_id: str):
        """Delete a conversation and all its history"""
        self.conversation_manager.delete_session(session_id)
    
    def rename_conversation(self, session_id: str, new_title: str):
        """Rename a conversation"""
        self.conversation_manager.rename_session(session_id, new_title)
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get all messages in a conversation"""
        messages = self.conversation_manager.get_messages(session_id)
        return [msg.to_dict() for msg in messages]
    
    # ========================================================================
    # CONVERSATIONAL INTERACTION
    # ========================================================================
    
    def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        auto_title: bool = True
    ) -> Dict:
        """
        Send a message and get a response.
        
        Args:
            user_message: User's question/request
            session_id: Conversation ID. Creates new if None.
            auto_title: Auto-generate title from first message
        
        Returns:
            {
                "session_id": str,
                "response": str,
                "visualizations": List[str],
                "tool_calls": List[Dict],
                "conversation_length": int
            }
        """
        
        # Create new session if needed
        if not session_id:
            title = generate_session_title(user_message) if auto_title else None
            session_id = self.create_conversation(title)
        
        # Get conversation history
        history = self.conversation_manager.messages_to_langgraph(session_id)
        
        # Add system prompt if first message
        if len(history) == 0:
            system_context = self.base_agent._build_system_prompt()
            history.insert(0, HumanMessage(content=system_context))
        
        # Add user message
        history.append(HumanMessage(content=user_message))
        
        # Run agent
        initial_state = {
            "messages": history,
            "insights_generated": 0,
            "visualizations_created": 0,
            "objective": user_message,
            "mode":"chat"
        }
        print(f"this is the state intially.  {initial_state}")
        
        final_state = self.graph.invoke(initial_state)
        new_messages = final_state["messages"][len(history):]  # Only new messages
        
        # Extract response and metadata
        response_text = ""
        visualizations = []
        tool_calls = []
        
        for msg in new_messages:
            # 1. IDENTIFY THE MESSAGE TYPE
            is_ai = isinstance(msg, AIMessage)
            is_tool = isinstance(msg, ToolMessage)

            # 2. EXTRACT TOOL CALLS (From AI Messages)
            # We check if the list actually has items
            if is_ai and msg.tool_calls:
                for call in msg.tool_calls:
                    tool_calls.append({
                        "name": call.get("name"),
                        "args": call.get("args", {}),
                        "id": call.get("id")
                    })

            # 3. EXTRACT VISUALS (From Tool Results)
            elif is_tool:
                if "viz_path" in msg.content:
                    try:
                        import json
                        result = json.loads(msg.content)
                        if "viz_path" in result:
                            visualizations.append(result["viz_path"])
                    except:
                        pass
                # IMPORTANT: We 'continue' here so Tool JSON never sets response_text
                continue 

            # 4. EXTRACT FINAL TEXT (From AI Messages with NO tool calls)
            elif is_ai and not msg.tool_calls:
                # Gemini 2.5 often sends content as a LIST of DICTS
                if isinstance(msg.content, list):
                    # Extract 'text' from each dictionary in the list
                    parts = []
                    for part in msg.content:
                        if isinstance(part, dict) and "text" in part:
                            parts.append(part["text"])
                        elif isinstance(part, str):
                            parts.append(part)
                    response_text = "\n".join(parts)
                else:
                    # Fallback for standard string content
                    response_text = msg.content
        
        # Save to conversation history
        self.conversation_manager.add_message(session_id, "human", user_message)
        self.conversation_manager.add_message(
            session_id,
            "ai",
            response_text,
            tool_calls=tool_calls if tool_calls else None,
            visualizations=visualizations if visualizations else None
        )
        
        return {
            "session_id": session_id,
            "response": response_text,
            "visualizations": visualizations,
            "tool_calls": tool_calls,
            "conversation_length": len(history) + len(new_messages)
        }
    
    def continue_conversation(
        self,
        session_id: str,
        user_message: str
    ) -> Dict:
        """
        Continue an existing conversation.
        Alias for chat() with explicit session_id.
        """
        return self.chat(user_message, session_id=session_id)
    
    # ========================================================================
    # CONVERSATION ANALYTICS
    # ========================================================================
    
    def get_conversation_stats(self, session_id: str) -> Dict:
        """Get statistics about a conversation"""
        return self.conversation_manager.get_session_stats(session_id)
    
    def export_conversation(
        self,
        session_id: str,
        format: str = "markdown"
    ) -> str:
        """
        Export conversation to a file format.
        
        Args:
            session_id: Conversation ID
            format: "json" or "markdown"
        
        Returns:
            Formatted string
        """
        return self.conversation_manager.export_conversation(session_id, format)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def search_conversations(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search conversations by content (simple text match).
        
        Args:
            query: Search term
            limit: Max results
        
        Returns:
            List of matching sessions with context
        """
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = cursor.execute("""
            SELECT DISTINCT s.session_id, s.title, s.last_updated,
                   m.content as matching_message
            FROM conversation_sessions s
            JOIN conversation_messages m ON s.session_id = m.session_id
            WHERE m.content LIKE ?
            ORDER BY s.last_updated DESC
            LIMIT ?
        """, (f"%{query}%", limit)).fetchall()
        
        conn.close()
        
        return [
            {
                "session_id": row[0],
                "title": row[1],
                "last_updated": row[2],
                "preview": row[3][:100] + "..." if len(row[3]) > 100 else row[3]
            }
            for row in results
        ]
    
    def get_recent_visualizations(self, session_id: str, limit: int = 5) -> List[str]:
        """Get most recent visualizations from a conversation"""
        stats = self.conversation_manager.get_session_stats(session_id)
        viz_paths = stats.get("visualization_paths", [])
        return viz_paths[-limit:] if viz_paths else []


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize agent
    agent = ConversationalEmailAgent()
    
    # Example 1: Start new conversation
    print("="*80)
    print("Example 1: New Conversation")
    print("="*80)
    
    result = agent.chat("Which brand discounts most aggressively?")
    print(f"Session ID: {result['session_id']}")
    print(f"Response: {result['response'][:200]}...")
    print(f"Visualizations: {result['visualizations']}")
    
    # Example 2: Continue conversation
    print("\n" + "="*80)
    print("Example 2: Continue Conversation")
    print("="*80)
    
    result2 = agent.continue_conversation(
        result['session_id'],
        "Show me a trend chart for that brand"
    )
    print(f"Response: {result2['response'][:200]}...")
    
    # Example 3: List all conversations
    print("\n" + "="*80)
    print("Example 3: List Conversations")
    print("="*80)
    
    conversations = agent.list_conversations()
    for conv in conversations[:3]:
        print(f"- {conv['title']} ({conv['message_count']} messages)")
    
    # Example 4: Export conversation
    print("\n" + "="*80)
    print("Example 4: Export Conversation")
    print("="*80)
    
    markdown = agent.export_conversation(result['session_id'], format="markdown")
    print(markdown[:300] + "...")