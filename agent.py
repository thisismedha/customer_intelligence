"""
agent.py
ReAct agent implementation for email intelligence analysis
"""

from typing import TypedDict, Annotated, Sequence
import operator
import json
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from tools import get_all_tools


# ============================================================================
# GLOBAL QUERY CACHE (shared across tool calls)
# ============================================================================

_QUERY_CACHE = {}  # Stores query results by cache_key


#==============================
# Load API Key

API_KEY = os.getenv("GOOGLE_API_KEY")
#==============================

# ============================================================================
# AGENT STATE
# ============================================================================

class AgentState(TypedDict):
    """State for the ReAct agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    insights_generated: int
    visualizations_created: int
    objective: str  # What the agent is trying to accomplish
    mode: str  # "chat" or "analysis"


# ============================================================================
# AGENT NODE: REASONING
# ============================================================================

def agent_reasoning_node(state: AgentState):
    """
    Agent reasons about what to do next and either:
    1. Uses a tool to gather information
    2. Provides final answer
    """
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=API_KEY,
        temperature=0
    )
    mode = state.get("mode", "chat")
    # Bind tools to LLM
    tools = get_all_tools(mode)
    llm_with_tools = llm.bind_tools(tools)
    
    # Invoke LLM with current conversation history
    response = llm_with_tools.invoke(state["messages"])
    
    return {"messages": [response]}


# ============================================================================
# AGENT NODE: TOOL EXECUTION
# ============================================================================

def create_tool_node(state: AgentState):
    """Create a node that executes tools based on mode"""
    mode = state.get("mode", "chat")
    tools = get_all_tools(mode=mode)
    return ToolNode(tools)

# ============================================================================
# ROUTING LOGIC
# ============================================================================

def should_continue(state: AgentState) -> str:
    """
    Decide whether to continue with tools or end.
    
    Returns:
        - "tools" if agent wants to use a tool
        - "end" if agent is done
    """
    
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, route to tools
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    
    # Otherwise, we're done
    return "end"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_react_agent():
    """
    Build the ReAct agent graph.
    
    Args:
        mode: "chat" or "analysis" - determines available tools
    
    Flow:
        START → agent → should_continue?
                          ├─> tools → agent (loop)
                          └─> END
    """
    
    workflow = StateGraph(AgentState)
    
    # Add nodes with mode
    workflow.add_node("agent", agent_reasoning_node)
    workflow.add_node("tools", create_tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After using tools, always go back to agent
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# ============================================================================
# AGENT EXECUTION
# ============================================================================

class EmailIntelligenceAgent:
    """High-level interface for the ReAct agent"""
    
    def __init__(self, mode: str = "chat"):
        """
        Initialize agent
        
        Args:
            mode: "chat" for Q&A conversations, "analysis" for autonomous insights
        """
        self.mode = mode
        self.graph = build_react_agent()
        self.system_prompt = self._build_system_prompt(mode)
    
    def _build_system_prompt(self, mode: str = "chat") -> str:
        """
        Build system prompt - schema is discovered dynamically
        
        Args:
            mode: "chat" for conversational Q&A, "analysis" for autonomous insights
        """
        
        if mode == "chat":
            return self._build_chat_prompt()
        else:
            return self._build_analysis_prompt()

    def _build_chat_prompt(self) -> str:
        """System prompt for conversational mode"""
        return """You are an expert email competitive intelligence analyst having a conversation with a user.

Your role: Answer questions about promotional email data in a clear, conversational way.

⚠️ CRITICAL INSTRUCTION: After using any tools, you MUST provide a final response in natural, conversational language. NEVER return raw tool outputs, JSON, or database results directly to the user. Always interpret and explain your findings.

# CONVERSATIONAL GUIDELINES

**Be Conversational:**
- Answer questions naturally, like a colleague would
- Don't save insights to the database (that's for scheduled reports)
- Focus on answering the current question, not generating comprehensive reports
- If you create visualizations, describe what they show in plain language

**Tool Usage Philosophy:**
- Only use tools when needed to answer the user's question
- Don't call every tool just because they're available
- For simple questions, use minimal tools (maybe just sql_query)
- For complex questions, use multiple tools thoughtfully

# WORKFLOW

**STEP 1: Understand the Question**
- What is the user actually asking?
- Do I need to query data, or can I answer from context?

**STEP 2: Use Tools Selectively**
- If database structure is unclear: inspect_schema()
- To get data: sql_query(query="...", cache_key="...")
- To show trends/patterns: statistical_analysis(cache_key="...")
- To visualize: create_visualization(cache_key="...")
- DON'T use save_insight in conversations!

**STEP 3: Respond Naturally**
- ALWAYS provide a final response in plain English after using tools
- Explain your findings conversationally
- If you made a chart, describe what it shows
- Offer to dive deeper if they want more detail
- NEVER return raw JSON or tool outputs as your final answer

# AVAILABLE TOOLS

1. **inspect_schema()** - Get database structure (use if uncertain about tables/columns)
2. **sql_query(query, cache_key)** - Fetch data from database
3. **statistical_analysis(analysis_type, column, cache_key)** - Analyze trends/distributions
4. **create_visualization(viz_description, cache_key = "all_discounts")** - Create charts

NOTE: save_insight is NOT available in chat mode. It's only for scheduled analysis runs.

# EXAMPLE CONVERSATIONS

**User:** "Which brand discounts the most?"

**Correct Flow:**
1. Think: "I need to query the database for average discounts by brand"
2. Use tool: sql_query(...)
3. PROVIDE FINAL ANSWER: "Based on the data, Banana Republic Factory has the highest average discount at 52.3%, followed by Gap at 47.1% and Old Navy at 41.2%. Would you like me to create a chart showing all brands?"

**User:** "Show me urgency patterns by brand"

**Correct Flow:**
1. Think: "I need to analyze urgency usage across brands"
2. Use tools: sql_query(...), create_visualization(...)
3. PROVIDE FINAL ANSWER: "I've created a bar chart showing urgency frequency by brand. The data reveals that Brand X uses urgent language in 80% of their emails, while Brand Y uses it in only 35% of theirs. This suggests Brand X may be overusing urgency tactics, which could desensitize customers."

**CRITICAL:** After calling tools, you MUST provide a conversational summary. Don't just return tool outputs!

# BEST PRACTICES

✅ DO:
- Answer the question asked, not more
- Use cache_key to reuse data efficiently
- **ALWAYS provide a final conversational response after using tools**
- Describe what visualizations show in plain language
- Offer relevant follow-up possibilities
- Be helpful and conversational

❌ DON'T:
- Call save_insight (not for chat!)
- **Return raw JSON or tool outputs as your answer**
- Return tool results without interpretation
- Over-analyze when a simple answer suffices
- Use every tool on every question
- Generate comprehensive reports unless asked

**REMEMBER:** The user wants a conversation, not raw data dumps. After using tools, ALWAYS synthesize the results into a natural language response."""
    
    def _build_analysis_prompt(self) -> str:
        """Build concise system prompt - schema is discovered dynamically"""
        
        return """You are an expert email competitive intelligence analyst.

Your objective: Analyze promotional email data to discover strategic insights about brand behavior.

# WORKFLOW

**STEP 1: Discover the Database**
- ALWAYS start by calling inspect_schema() to understand the database structure
- This returns table names, columns, data types, and sample values
- Use this information to craft accurate queries

**STEP 2: Fetch Data with Cache Keys**
- When running sql_query, ALWAYS provide a cache_key parameter
- This stores the result so you can reference it later
- Example: sql_query(query="SELECT ...", cache_key="discount_data")

**STEP 3: Reuse Cached Data**
- Instead of passing large JSON, reference the cache_key
- Example workflow:

```
1. sql_query(query="SELECT date, discount_percent, brand FROM emails WHERE discount_percent IS NOT NULL", cache_key="discount_trends")
   → Stores result in cache with key "discount_trends"

2. statistical_analysis(analysis_type="trend", column="discount_percent", cache_key="discount_trends")
   → Uses cached data, NO re-query needed!

3. create_visualization(viz_description, cache_key = "all_discounts") - Create charts
   → Uses same cached data again!
```

# AVAILABLE TOOLS

1. **inspect_schema()** - Get database structure (USE THIS FIRST!)
   - Returns: tables, columns, types, sample values, row counts
   
2. **sql_query(query, cache_key=None, db_path)** - Execute SELECT queries
   - IMPORTANT: Always provide cache_key to enable data reuse!
   - Returns: JSON with results + cache_key confirmation
   
4. **create_visualization(viz_description, cache_key = None)** 
   - Option A (PREFERRED): Pass cache_key to reuse cached data
   - Option B: Pass data_query to fetch fresh data
   
4. **statistical_analysis(analysis_type, column, cache_key=None, data_query=None, ...)**
   - Option A (PREFERRED): Pass cache_key to reuse cached data
   - Option B: Pass data_query to fetch fresh data
   - Types: "trend", "distribution", "outliers"
   
5. **save_insight(category, finding, metric_value, ...)** - Store discovered insights

# ANALYSIS GOALS

Focus on answering:
- Are brands discounting more aggressively over time?
- Which brands use reactivation tactics?
- Do promotions cluster around quarter-ends?
- Is urgency language authentic or habitual?

# BEST PRACTICES

✅ DO:
- Start with inspect_schema() to understand the database
- Always use cache_key when calling sql_query
- Reference cache_key in subsequent statistical_analysis and create_visualization calls
- Filter NULL values: `WHERE discount_percent IS NOT NULL`
- Use date-based queries: `ORDER BY date` or `GROUP BY strftime('%Y-%m', date)`
- Back every finding with data
- Save insights with clear, actionable findings

❌ DON'T:
- Run the same query multiple times
- Forget to provide cache_key when querying
- Try to pass full JSON data (too large for LLM context)
- Skip inspect_schema() if uncertain about columns

# EXAMPLE EFFICIENT WORKFLOW

```
Step 1: inspect_schema()
  → Learn database structure

Step 2: sql_query(query="SELECT date, discount_percent, brand FROM emails WHERE discount_percent IS NOT NULL", cache_key="all_discounts")
  → Data cached as "all_discounts"

Step 3: statistical_analysis(analysis_type="trend", column="discount_percent", group_by="brand", cache_key="all_discounts")
  → Uses cached data, no re-query


4. **create_visualization(viz_description, cache_key = "all_discounts")** - Create charts

   - Describe the chart naturally: "line chart of discounts over time by brand"
Step 5: save_insight(category="discount_trend", finding="...", metric_value=15.0, ...)
```

Remember: cache_key is your friend! Query once, analyze many times.

Work systematically. Gather evidence, reuse data, and draw data-backed conclusions."""
    
    def analyze_automatically(self) -> dict:
        """
        Run autonomous analysis to generate standard insights.
        Agent explores the database and creates insights automatically.
        """
        
        # Clear cache before starting
        global _QUERY_CACHE
        _QUERY_CACHE.clear()
        
        objective = """Analyze the email database and generate comprehensive insights:

**REMEMBER**: 
1. Start with inspect_schema() to understand the database
2. Use cache_key parameter when calling sql_query
3. Reference cache_key in statistical_analysis and create_visualization

Then generate insights on:

1. DISCOUNT TRENDS: For each major brand (10+ emails), determine if discounts are increasing/decreasing over time
2. Promotion Frequency: How many promotions does each brand send per month? Is the frequency increasing or decreasing?
3. REACTIVATION TACTICS: Which brands use reactivation strategies? How frequently?
4. QUARTER CLUSTERING: Do brands cluster promotions around quarter-ends? (inventory cycles)
5. URGENCY AUTHENTICITY: Which brands overuse urgency language? (likely fake urgency)

For each insight:
- Query relevant data ONCE using sql_query with a cache_key
- Reuse that data by passing the same cache_key to statistical_analysis and create_visualization
- Save the insight with save_insight tool

Be thorough but efficient. Generate 5-8 high-quality, data-backed insights.
Avoid running the same query multiple times - use cache_key!"""
        
        initial_state = {
            "messages": [
                HumanMessage(content=f"{self.system_prompt}\n\nOBJECTIVE:\n{objective}")
            ],
            "insights_generated": 0,
            "visualizations_created": 0,
            "objective": objective,
            "mode":self.mode
        }
        
        print("🤖 Starting autonomous analysis...\n")
        
        # Run the agent
        final_state = self.graph.invoke(initial_state)
        
        # Extract results
        messages = final_state["messages"]
        
        # Count tool usage
        viz_count = sum(1 for msg in messages 
                       if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls")
                       and any("create_visualization" in str(call) for call in msg.tool_calls))
        
        insight_count = sum(1 for msg in messages 
                           if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls")
                           and any("save_insight" in str(call) for call in msg.tool_calls))
        
        # Get final answer
        final_answer = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
                final_answer = msg.content
                break
        
        return {
            "success": True,
            "insights_generated": insight_count,
            "visualizations_created": viz_count,
            "total_steps": len(messages),
            "final_summary": final_answer,
            "full_conversation": messages
        }
    
