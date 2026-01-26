"""
conversation_manager.py
Manages multi-session conversations with persistent history
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ConversationMetadata:
    """Metadata for a conversation session"""
    session_id: str
    title: str
    created_at: str
    last_updated: str
    message_count: int
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ConversationMessage:
    """Individual message in a conversation"""
    session_id: str
    message_id: str
    role: str  # "human", "ai", "tool"
    content: str
    tool_calls: Optional[str] = None  # JSON string
    visualizations: Optional[str] = None  # JSON array of paths
    timestamp: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """
    Manages conversation sessions and history.
    
    Features:
    - Create/resume/delete conversations
    - Persistent message history
    - Multi-session support
    - Visualization tracking
    """
    
    def __init__(self, db_path: str = "email_intel.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        """Create conversation tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversation sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0
            )
        """)
        
        # Conversation messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT,
                visualizations TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
            )
        """)
        
        # Index for fast session lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON conversation_messages(session_id, timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================
    
    def create_session(self, title: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            title: Optional custom title. Auto-generated if None.
        
        Returns:
            session_id
        """
        session_id = str(uuid.uuid4())
        
        if not title:
            title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_sessions (session_id, title)
            VALUES (?, ?)
        """, (session_id, title))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationMetadata]:
        """Get session metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        row = cursor.execute("""
            SELECT session_id, title, created_at, last_updated, message_count
            FROM conversation_sessions
            WHERE session_id = ?
        """, (session_id,)).fetchone()
        
        conn.close()
        
        if row:
            return ConversationMetadata(*row)
        return None
    
    def list_sessions(self, limit: int = 50) -> List[ConversationMetadata]:
        """Get all conversation sessions, most recent first"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        rows = cursor.execute("""
            SELECT session_id, title, created_at, last_updated, message_count
            FROM conversation_sessions
            ORDER BY last_updated DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
        conn.close()
        
        return [ConversationMetadata(*row) for row in rows]
    
    def delete_session(self, session_id: str):
        """Delete a conversation session and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM conversation_sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    
    def rename_session(self, session_id: str, new_title: str):
        """Rename a conversation session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE conversation_sessions 
            SET title = ?, last_updated = datetime('now')
            WHERE session_id = ?
        """, (new_title, session_id))
        
        conn.commit()
        conn.close()
    
    # ========================================================================
    # MESSAGE MANAGEMENT
    # ========================================================================
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        visualizations: Optional[List[str]] = None
    ) -> str:
        """
        Add a message to a conversation.
        
        Args:
            session_id: Conversation ID
            role: "human", "ai", or "tool"
            content: Message text
            tool_calls: Optional list of tool calls
            visualizations: Optional list of visualization paths
        
        Returns:
            message_id
        """
        message_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_messages 
            (message_id, session_id, role, content, tool_calls, visualizations)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            message_id,
            session_id,
            role,
            content,
            json.dumps(tool_calls) if tool_calls else None,
            json.dumps(visualizations) if visualizations else None
        ))
        
        # Update session metadata
        cursor.execute("""
            UPDATE conversation_sessions 
            SET message_count = message_count + 1,
                last_updated = datetime('now')
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        """
        Get all messages for a session.
        
        Args:
            session_id: Conversation ID
            limit: Optional max messages to return (most recent)
        
        Returns:
            List of messages in chronological order
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT message_id, session_id, role, content, 
                   tool_calls, visualizations, timestamp
            FROM conversation_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        rows = cursor.execute(query, (session_id,)).fetchall()
        conn.close()
        
        return [ConversationMessage(*row) for row in rows]
    
    def clear_session_messages(self, session_id: str):
        """Clear all messages from a session (keep session metadata)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
        cursor.execute("""
            UPDATE conversation_sessions 
            SET message_count = 0, last_updated = datetime('now')
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    # ========================================================================
    # LANGGRAPH INTEGRATION
    # ========================================================================
    
    def messages_to_langgraph(
        self,
        session_id: str
    ) -> List[BaseMessage]:
        """
        Convert stored messages to LangGraph format for agent resumption.
        
        Returns:
            List of BaseMessage objects (HumanMessage, AIMessage, ToolMessage)
        """
        messages = self.get_messages(session_id)
        langgraph_messages = []
        
        for msg in messages:
            if msg.role == "human":
                langgraph_messages.append(HumanMessage(content=msg.content))
            
            elif msg.role == "ai":
                ai_msg = AIMessage(content=msg.content)
                
                # Add tool calls if present - ensure they have proper structure with 'id'
                if msg.tool_calls:
                    tool_calls = json.loads(msg.tool_calls)
                    # Ensure each tool call has an 'id' field
                    for i, call in enumerate(tool_calls):
                        if 'id' not in call:
                            call['id'] = f"{msg.message_id}_{i}"
                    ai_msg.tool_calls = tool_calls
                
                langgraph_messages.append(ai_msg)
            
            elif msg.role == "tool":
                langgraph_messages.append(ToolMessage(
                    content=msg.content,
                    tool_call_id=msg.message_id
                ))
        
        return langgraph_messages
    
    def save_langgraph_messages(
        self,
        session_id: str,
        messages: List[BaseMessage],
        extract_visualizations: bool = True
    ):
        """
        Save LangGraph messages to conversation history.
        
        Args:
            session_id: Conversation ID
            messages: List of LangGraph messages
            extract_visualizations: Parse tool responses for viz paths
        """
        for msg in messages:
            if isinstance(msg, HumanMessage):
                self.add_message(session_id, "human", msg.content)
            
            elif isinstance(msg, AIMessage):
                tool_calls = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = [
                        {
                            "name": call.get("name"),
                            "args": call.get("args", {})
                        }
                        for call in msg.tool_calls
                    ]
                
                self.add_message(
                    session_id,
                    "ai",
                    msg.content,
                    tool_calls=tool_calls
                )
            
            elif isinstance(msg, ToolMessage) and extract_visualizations:
                # Extract visualization paths from tool responses
                viz_paths = []
                try:
                    result = json.loads(msg.content)
                    if "viz_path" in result:
                        viz_paths.append(result["viz_path"])
                except:
                    pass
                
                self.add_message(
                    session_id,
                    "tool",
                    msg.content,
                    visualizations=viz_paths if viz_paths else None
                )
    
    # ========================================================================
    # ANALYTICS
    # ========================================================================
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a conversation session"""
        messages = self.get_messages(session_id)
        
        visualizations = []
        for msg in messages:
            if msg.visualizations:
                visualizations.extend(json.loads(msg.visualizations))
        
        tool_usage = {}
        for msg in messages:
            if msg.role == "ai" and msg.tool_calls:
                calls = json.loads(msg.tool_calls)
                for call in calls:
                    tool_name = call.get("name", "unknown")
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        return {
            "total_messages": len(messages),
            "human_messages": sum(1 for m in messages if m.role == "human"),
            "ai_messages": sum(1 for m in messages if m.role == "ai"),
            "visualizations_created": len(visualizations),
            "tool_usage": tool_usage,
            "visualization_paths": visualizations
        }
    
    def export_conversation(self, session_id: str, format: str = "json") -> str:
        """
        Export conversation to various formats.
        
        Args:
            session_id: Conversation ID
            format: "json" or "markdown"
        
        Returns:
            Formatted string
        """
        session = self.get_session(session_id)
        messages = self.get_messages(session_id)
        
        if format == "json":
            return json.dumps({
                "session": session.to_dict(),
                "messages": [msg.to_dict() for msg in messages]
            }, indent=2)
        
        elif format == "markdown":
            lines = [
                f"# {session.title}",
                f"Created: {session.created_at}",
                f"Messages: {session.message_count}",
                "",
                "---",
                ""
            ]
            
            for msg in messages:
                if msg.role == "human":
                    lines.append(f"## 👤 User")
                    lines.append(msg.content)
                elif msg.role == "ai":
                    lines.append(f"## 🤖 Assistant")
                    lines.append(msg.content)
                    if msg.visualizations:
                        viz = json.loads(msg.visualizations)
                        for path in viz:
                            lines.append(f"![Visualization]({path})")
                
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unknown format: {format}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_session_title(first_message: str, max_length: int = 50) -> str:
    """Generate a descriptive title from the first user message"""
    
    # Take first sentence or up to max_length
    title = first_message.split('.')[0].strip()
    
    if len(title) > max_length:
        title = title[:max_length].rsplit(' ', 1)[0] + "..."
    
    return title or "New Conversation"