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
    
    # Bind tools to LLM
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    # Invoke LLM with current conversation history
    response = llm_with_tools.invoke(state["messages"])
    
    return {"messages": [response]}


# ============================================================================
# AGENT NODE: TOOL EXECUTION
# ============================================================================

def create_tool_node():
    """Create a node that executes tools"""
    tools = get_all_tools()
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
    
    Flow:
        START → agent → should_continue?
                          ├─> tools → agent (loop)
                          └─> END
    """
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_reasoning_node)
    workflow.add_node("tools", create_tool_node())
    
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
    
    def __init__(self):
        self.graph = build_react_agent()
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
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

3. create_visualization(viz_type="line", x_column="date", y_column="discount_percent", cache_key="discount_trends")
   → Uses same cached data again!
```

# AVAILABLE TOOLS

1. **inspect_schema()** - Get database structure (USE THIS FIRST!)
   - Returns: tables, columns, types, sample values, row counts
   
2. **sql_query(query, cache_key=None, db_path)** - Execute SELECT queries
   - IMPORTANT: Always provide cache_key to enable data reuse!
   - Returns: JSON with results + cache_key confirmation
   
3. **create_visualization(viz_type, title, x_column, y_column, cache_key=None, data_query=None, ...)**
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

Step 4: create_visualization(viz_type="line", x_column="date", y_column="discount_percent", color_column="brand", cache_key="all_discounts")
  → Uses same cached data

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

Be thorough but efficient. Generate 8-12 high-quality, data-backed insights.
Avoid running the same query multiple times - use cache_key!"""
        
        initial_state = {
            "messages": [
                HumanMessage(content=f"{self.system_prompt}\n\nOBJECTIVE:\n{objective}")
            ],
            "insights_generated": 0,
            "visualizations_created": 0,
            "objective": objective
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
    
    def answer_question(self, question: str) -> dict:
        """
        Answer a specific user question using the agent.
        
        Args:
            question: User's question about the email data
            
        Returns:
            dict with answer and any generated artifacts
        """
        
        # Clear cache before starting
        global _QUERY_CACHE
        _QUERY_CACHE.clear()
        
        initial_state = {
            "messages": [
                HumanMessage(content=f"{self.system_prompt}\n\nUser Question: {question}")
            ],
            "insights_generated": 0,
            "visualizations_created": 0,
            "objective": question
        }
        
        print(f"🤖 Processing question: {question}\n")
        
        final_state = self.graph.invoke(initial_state)
        messages = final_state["messages"]
        
        # Extract answer
        answer = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
                answer = msg.content
                break
        
        # Extract visualizations created
        viz_paths = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and "viz_path" in msg.content:
                try:
                    result = json.loads(msg.content)
                    if "viz_path" in result:
                        viz_paths.append(result["viz_path"])
                except:
                    pass
        
        return {
            "question": question,
            "answer": answer,
            "visualizations": viz_paths,
            "steps_taken": len(messages)
        }


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_agent_conversation(messages: Sequence[BaseMessage], max_content_len: int = 300):
    """Pretty print the agent's reasoning process"""
    
    print("\n" + "="*80)
    print("🧠 AGENT REASONING TRACE")
    print("="*80 + "\n")
    
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, HumanMessage):
            print(f"👤 HUMAN (Step {i}):")
            print(f"   {msg.content[:max_content_len]}...")
            
        elif isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
                print(f"\n🤖 AGENT REASONING (Step {i}):")
                if msg.content:
                    print(f"   Thought: {msg.content[:max_content_len]}")
                
                for tool_call in msg.tool_calls:
                    print(f"\n   🔧 Action: {tool_call['name']}")
                    args = tool_call.get('args', {})
                    for key, value in args.items():
                        val_str = str(value)[:100]
                        print(f"      - {key}: {val_str}")
            else:
                print(f"\n✅ AGENT FINAL ANSWER (Step {i}):")
                print(f"   {msg.content[:max_content_len]}...")
                
        elif isinstance(msg, ToolMessage):
            print(f"\n   📊 Observation:")
            try:
                result = json.loads(msg.content)
                if "error" in result:
                    print(f"      ❌ Error: {result['error']}")
                else:
                    # Pretty print key results
                    if "rows" in result:
                        print(f"      ✓ Retrieved {result['rows']} rows")
                    if "cache_key" in result:
                        print(f"      ✓ Cached as: '{result['cache_key']}'")
                    if "viz_path" in result:
                        print(f"      ✓ Created: {result['viz_path']}")
                    if "success" in result:
                        print(f"      ✓ Success")
                    if "tables" in result:
                        print(f"      ✓ Found {len(result['tables'])} tables")
            except:
                print(f"      {msg.content[:200]}")
        
        print()
    
    print("="*80 + "\n")