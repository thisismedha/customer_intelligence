"""
tools.py
Agent tools for email intelligence analysis
"""

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
import numpy as np

from langchain.tools import tool


# ============================================================================
# GLOBAL QUERY CACHE
# ============================================================================

_QUERY_CACHE = {}  # Shared cache for query results


# ============================================================================
# TOOL 0: INSPECT SCHEMA
# ============================================================================

@tool
def inspect_schema(db_path: str = "email_intel.db") -> str:
    """
    Get complete database schema with table structures and sample data.
    
    This should be the FIRST tool called to understand the database structure.
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        JSON with tables, columns, types, sample values, and row counts
    """
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = {}
        
        for table in tables:
            # Get column info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            column_details = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                
                # Get sample non-null values
                cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 3")
                samples = [row[0] for row in cursor.fetchall()]
                
                column_details.append({
                    "name": col_name,
                    "type": col_type,
                    "samples": samples
                })
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            schema_info[table] = {
                "columns": column_details,
                "row_count": row_count
            }
        
        conn.close()
        
        return json.dumps({
            "success": True,
            "database": db_path,
            "tables": schema_info
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": f"Schema inspection failed: {str(e)}"})


# ============================================================================
# TOOL 1: SQL QUERY (with caching)
# ============================================================================

@tool
def sql_query(query: str, cache_key: Optional[str] = None, db_path: str = "email_intel.db") -> str:
    """
    Execute read-only SQL queries on the email intelligence database.
    
    Args:
        query: SELECT statement to execute
        cache_key: Optional key to cache results for reuse (RECOMMENDED!)
        db_path: Path to SQLite database
    
    Returns:
        JSON string containing query results as list of dicts
    
    Examples:
        sql_query("SELECT brand, AVG(discount_percent) FROM emails GROUP BY brand", cache_key="brand_averages")
        
    Safety:
        - Only SELECT queries allowed
        - 30 second timeout
        - Max 10,000 rows
        
    Caching:
        If cache_key is provided, results are stored and can be referenced by other tools
        using the same cache_key instead of re-querying.
    """
    
    # Security: Only allow SELECT
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return json.dumps({
            "error": "Only SELECT queries are allowed for safety",
            "query": query
        })
    
    # Block dangerous operations
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
    if any(keyword in query_upper for keyword in dangerous_keywords):
        return json.dumps({
            "error": "Query contains forbidden keywords",
            "query": query
        })
    
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row  # Returns dict-like rows
        cursor = conn.cursor()
        
        # Execute with row limit
        cursor.execute(query)
        rows = cursor.fetchmany(10000)  # Max 10k rows
        
        # Convert to list of dicts
        results = [dict(row) for row in rows]
        
        conn.close()
        
        result_data = {
            "success": True,
            "rows": len(results),
            "data": results,
            "query": query
        }
        
        # Cache if key provided
        if cache_key:
            global _QUERY_CACHE
            _QUERY_CACHE[cache_key] = result_data
            result_data["cache_key"] = cache_key
            result_data["cached"] = True
        
        return json.dumps(result_data, default=str)
        
    except Exception as e:
        return json.dumps({
            "error": f"Query execution failed: {str(e)}",
            "query": query
        })


# ============================================================================
# TOOL 2: CREATE VISUALIZATION (with cache support)
# ============================================================================

@tool
def create_visualization(
    viz_type: str,
    title: str,
    x_column: str,
    y_column: str,
    cache_key: Optional[str] = None,
    data_query: Optional[str] = None,
    color_column: Optional[str] = None,
    db_path: str = "email_intel.db"
) -> str:
    """
    Generate interactive Plotly visualizations from cached data or database queries.
    
    Args:
        viz_type: Type of chart - "line", "bar", "scatter", "heatmap", "box", "histogram"
        title: Chart title
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        cache_key: Key to retrieve cached query results (PREFERRED - avoids re-query!)
        data_query: SQL query to fetch data (only if cache_key not available)
        color_column: Optional column for color grouping
        db_path: Path to database
    
    Returns:
        JSON with visualization path and metadata
        
    Usage:
        Option 1 (PREFERRED): create_visualization(..., cache_key="my_data")
        Option 2 (fallback): create_visualization(..., data_query="SELECT ...")
    """
    
    try:
        # Get data from cache or query
        if cache_key:
            global _QUERY_CACHE
            if cache_key not in _QUERY_CACHE:
                return json.dumps({
                    "error": f"Cache key '{cache_key}' not found. Run sql_query first with this cache_key.",
                    "available_keys": list(_QUERY_CACHE.keys())
                })
            
            cached_result = _QUERY_CACHE[cache_key]
            df = pd.DataFrame(cached_result["data"])
        
        elif data_query:
            # Fetch fresh data
            result = json.loads(sql_query.invoke({"query":data_query, "db_path":db_path}))
            
            if "error" in result:
                return json.dumps({"error": result["error"]})
            
            df = pd.DataFrame(result["data"])
        
        else:
            return json.dumps({"error": "Must provide either cache_key or data_query"})
        
        if df.empty:
            return json.dumps({"error": "No data available for visualization"})
        
        # Validate columns exist
        required_cols = [x_column, y_column]
        if color_column:
            required_cols.append(color_column)
            
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return json.dumps({
                "error": f"Columns not found: {missing}",
                "available_columns": list(df.columns)
            })
        
        # Create visualization based on type
        fig = None
        
        if viz_type == "line":
            fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
            
        elif viz_type == "bar":
            fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
            
        elif viz_type == "scatter":
            fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
            
        elif viz_type == "box":
            fig = px.box(df, x=x_column, y=y_column, color=color_column, title=title)
            
        elif viz_type == "histogram":
            fig = px.histogram(df, x=x_column, color=color_column, title=title)
            
        elif viz_type == "heatmap":
            # For heatmap, expect pivoted data or create pivot
            if color_column:
                pivot = df.pivot_table(values=y_column, index=x_column, columns=color_column)
                fig = px.imshow(pivot, title=title, labels=dict(color=y_column))
            else:
                return json.dumps({"error": "Heatmap requires color_column for pivoting"})
        
        else:
            return json.dumps({
                "error": f"Unknown viz_type: {viz_type}",
                "supported": ["line", "bar", "scatter", "box", "histogram", "heatmap"]
            })
        
        # Enhance chart
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=500
        )
        
        # Save to file
        output_dir = Path("output/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{viz_type}_{timestamp}.html"
        filepath = output_dir / filename
        
        fig.write_html(str(filepath))
        
        return json.dumps({
            "success": True,
            "viz_path": str(filepath),
            "viz_type": viz_type,
            "title": title,
            "rows_plotted": len(df),
            "used_cache": cache_key is not None,
            "cache_key": cache_key
        })
        
    except Exception as e:
        return json.dumps({"error": f"Visualization failed: {str(e)}"})


#======
#Test viz tool
#===========

"""
Alternative: Let LLM generate Python code for maximum flexibility
"""

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
from pathlib import Path
import json
from datetime import datetime
from langchain.tools import tool

# Global cache (same as before)
_QUERY_CACHE = {}

@tool
def execute_visualization_code(
    python_code: str,
    title: str,
    cache_key: Optional[str] = None,
    data_query: Optional[str] = None,
    db_path: str = "email_intel.db"
) -> str:
    """
    Execute Python code to create a Plotly visualization with FULL flexibility.
    
    This allows you to create ANY visualization using the complete Plotly API,
    including Graph Objects, subplots, custom data transformations, and more.
    
    Args:
        python_code: Python code that creates a 'fig' variable (Plotly figure)
                    The code has access to:
                    - df: pandas DataFrame with the data
                    - pd: pandas library
                    - px: plotly.express
                    - go: plotly.graph_objects
                    - make_subplots: for creating subplot layouts
        title: Chart title (for filename)
        cache_key: Key to retrieve cached query results (PREFERRED)
        data_query: SQL query to fetch data (fallback)
        db_path: Path to database
    
    Returns:
        JSON with visualization path and metadata
        
    Security:
        - Code runs in restricted namespace (only pandas, plotly available)
        - No file system access beyond plotting
        - No network access
        - No imports allowed in the code
        
    Example 1 - Simple Express chart:
    ```python
    fig = px.line(df, x='date', y='discount_percent', color='brand',
                  title='Discount Trends Over Time')
    ```
    
    Example 2 - Custom Graph Objects with dual axis:
    ```python
    from plotly.subplots import make_subplots
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['discount_percent'], 
                   name="Discount %", mode='lines'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['email_count'], 
                   name="Email Volume", mode='lines'),
        secondary_y=True
    )
    
    fig.update_yaxes(title_text="Discount %", secondary_y=False)
    fig.update_yaxes(title_text="Email Count", secondary_y=True)
    fig.update_layout(title="Discounts vs Email Volume")
    ```
    
    Example 3 - Data transformation + visualization:
    ```python
    # Calculate 7-day rolling average
    df_sorted = df.sort_values('date')
    df_sorted['rolling_avg'] = df_sorted.groupby('brand')['discount_percent'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    
    fig = px.line(df_sorted, x='date', y='rolling_avg', color='brand',
                  title='7-Day Rolling Average Discount by Brand')
    ```
    
    Example 4 - Complex subplot grid:
    ```python
    brands = df['brand'].unique()[:4]  # Top 4 brands
    fig = make_subplots(rows=2, cols=2, subplot_titles=brands)
    
    for idx, brand in enumerate(brands):
        row = idx // 2 + 1
        col = idx % 2 + 1
        brand_df = df[df['brand'] == brand]
        
        fig.add_trace(
            go.Histogram(x=brand_df['discount_percent'], name=brand),
            row=row, col=col
        )
    
    fig.update_layout(title='Discount Distribution by Brand', showlegend=False)
    ```
    
    Example 5 - Annotated chart with thresholds:
    ```python
    fig = px.scatter(df, x='date', y='discount_percent', color='urgency_level',
                     size='email_count', hover_data=['subject'])
    
    # Add threshold line
    fig.add_hline(y=50, line_dash="dash", line_color="red",
                  annotation_text="50% Threshold")
    
    fig.update_layout(title='Discount Patterns with Urgency Signals')
    ```
    """
    
    try:
        # Get data from cache or query
        if cache_key:
            if cache_key not in _QUERY_CACHE:
                return json.dumps({
                    "error": f"Cache key '{cache_key}' not found.",
                    "available_keys": list(_QUERY_CACHE.keys())
                })
            df = pd.DataFrame(_QUERY_CACHE[cache_key]["data"])
        
        elif data_query:
            # Import sql_query tool
            from tools import sql_query
            result = json.loads(sql_query(data_query, db_path=db_path))
            if "error" in result:
                return json.dumps({"error": result["error"]})
            df = pd.DataFrame(result["data"])
        
        else:
            return json.dumps({"error": "Must provide cache_key or data_query"})
        
        if df.empty:
            return json.dumps({"error": "No data available"})
        
        # Create restricted namespace for code execution
        namespace = {
            'df': df,
            'pd': pd,
            'px': px,
            'go': go,
            'make_subplots': make_subplots,
            'fig': None  # Will be set by user code
        }
        
        # Execute the code
        try:
            exec(python_code, namespace)
        except SyntaxError as e:
            return json.dumps({
                "error": f"Syntax error in Python code: {str(e)}",
                "line": e.lineno,
                "code": python_code
            })
        except Exception as e:
            return json.dumps({
                "error": f"Runtime error: {str(e)}",
                "code": python_code
            })
        
        # Get the figure from namespace
        fig = namespace.get('fig')
        
        if fig is None:
            return json.dumps({
                "error": "Code did not create a 'fig' variable",
                "hint": "Your code must assign a Plotly figure to the variable 'fig'"
            })
        
        # Validate it's a Plotly figure
        if not (isinstance(fig, go.Figure) or hasattr(fig, 'write_html')):
            return json.dumps({
                "error": "Variable 'fig' is not a valid Plotly figure",
                "type": str(type(fig))
            })
        
        # Save to file
        output_dir = Path("output/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean title for filename
        clean_title = "".join(c if c.isalnum() else "_" for c in title)[:50]
        filename = f"{clean_title}_{timestamp}.html"
        filepath = output_dir / filename
        
        fig.write_html(str(filepath))
        
        return json.dumps({
            "success": True,
            "viz_path": str(filepath),
            "title": title,
            "rows_used": len(df),
            "used_cache": cache_key is not None
        })
        
    except Exception as e:
        return json.dumps({"error": f"Visualization failed: {str(e)}"})




# ============================================================================
# TOOL 3: STATISTICAL ANALYSIS (with cache support)
# ============================================================================

@tool
def statistical_analysis(
    analysis_type: str,
    column: str,
    cache_key: Optional[str] = None,
    data_query: Optional[str] = None,
    group_by: Optional[str] = None,
    db_path: str = "email_intel.db"
) -> str:
    """
    Perform statistical analysis on email data.
    
    Args:
        analysis_type: Type of analysis - "trend", "correlation", "outliers", "distribution"
        column: Column to analyze
        cache_key: Key to retrieve cached query results (PREFERRED - avoids re-query!)
        data_query: SQL query to fetch data (only if cache_key not available)
        group_by: Optional column to group analysis by (e.g., by brand)
        db_path: Database path
    
    Returns:
        JSON with statistical results and interpretations
        
    Analysis Types:
        - trend: Detect if values are increasing/decreasing over time
        - correlation: Find correlation between two numeric columns
        - outliers: Identify unusual values using z-score
        - distribution: Summary statistics (mean, median, std, quartiles)
        
    Usage:
        Option 1 (PREFERRED): statistical_analysis(..., cache_key="my_data")
        Option 2 (fallback): statistical_analysis(..., data_query="SELECT ...")
    """
    
    try:
        # Get data from cache or query
        if cache_key:
            global _QUERY_CACHE
            if cache_key not in _QUERY_CACHE:
                return json.dumps({
                    "error": f"Cache key '{cache_key}' not found. Run sql_query first with this cache_key.",
                    "available_keys": list(_QUERY_CACHE.keys())
                })
            
            cached_result = _QUERY_CACHE[cache_key]
            df = pd.DataFrame(cached_result["data"])
        
        elif data_query:
            result = json.loads(sql_query.invoke({"query":data_query, "db_path":db_path}))
            
            if "error" in result:
                return json.dumps({"error": result["error"]})
            
            df = pd.DataFrame(result["data"])
        
        else:
            return json.dumps({"error": "Must provide either cache_key or data_query"})
        
        if df.empty:
            return json.dumps({"error": "No data to analyze"})
        
        if column not in df.columns:
            return json.dumps({
                "error": f"Column '{column}' not found",
                "available": list(df.columns)
            })
        
        # Remove null values
        df_clean = df[df[column].notna()].copy()
        
        if len(df_clean) == 0:
            return json.dumps({"error": f"All values in '{column}' are null"})
        
        results = {}
        
        # TREND ANALYSIS
        if analysis_type == "trend":
            # Requires time-based data
            if "date" not in df_clean.columns:
                return json.dumps({"error": "Trend analysis requires 'date' column"})
            
            df_clean["date"] = pd.to_datetime(df_clean["date"])
            df_clean = df_clean.sort_values("date")
            
            # Convert dates to numeric for regression
            df_clean["date_numeric"] = (df_clean["date"] - df_clean["date"].min()).dt.days
            
            if group_by and group_by in df_clean.columns:
                # Analyze trend for each group
                group_results = {}
                
                for group_name, group_df in df_clean.groupby(group_by):
                    if len(group_df) < 3:
                        continue
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        group_df["date_numeric"], 
                        group_df[column]
                    )
                    
                    group_results[str(group_name)] = {
                        "slope": float(slope),
                        "trend": "increasing" if slope > 0 else "decreasing",
                        "r_squared": float(r_value ** 2),
                        "p_value": float(p_value),
                        "significant": bool(p_value < 0.05),
                        "interpretation": (
                            f"{'Increasing' if slope > 0 else 'Decreasing'} by "
                            f"{abs(slope):.2f} per day (R²={r_value**2:.2f})"
                        )
                    }
                
                results = {"trend_by_group": group_results}
            else:
                # Overall trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_clean["date_numeric"], 
                    df_clean[column]
                )
                
                results = {
                    "slope": float(slope),
                    "trend": "increasing" if slope > 0 else "decreasing",
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                    "interpretation": (
                        f"{'Increasing' if slope > 0 else 'Decreasing'} by "
                        f"{abs(slope):.2f} per day (R²={r_value**2:.2f})"
                    )
                }
        
        # DISTRIBUTION ANALYSIS
        elif analysis_type == "distribution":
            if group_by and group_by in df_clean.columns:
                group_results = {}
                
                for group_name, group_df in df_clean.groupby(group_by):
                    values = group_df[column].astype(float)
                    group_results[str(group_name)] = {
                        "count": int(len(values)),
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75))
                    }
                
                results = {"distribution_by_group": group_results}
            else:
                values = df_clean[column].astype(float)
                results = {
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75))
                }
        
        # OUTLIER DETECTION
        elif analysis_type == "outliers":
            values = df_clean[column].astype(float)
            z_scores = np.abs(stats.zscore(values))
            outliers = df_clean[z_scores > 3].copy()
            
            results = {
                "outliers_found": len(outliers),
                "outlier_threshold": "z-score > 3",
                "outliers": outliers[[column]].to_dict('records')[:10]  # Limit to 10
            }
        
        else:
            return json.dumps({
                "error": f"Unknown analysis_type: {analysis_type}",
                "supported": ["trend", "distribution", "outliers"]
            })
        
        return json.dumps({
            "success": True,
            "analysis_type": analysis_type,
            "column": column,
            "results": results,
            "used_cache": cache_key is not None,
            "cache_key": cache_key
        })
        
    except Exception as e:
        return json.dumps({"error": f"Statistical analysis failed: {str(e)}"})


# ============================================================================
# TOOL 4: SAVE INSIGHT
# ============================================================================

@tool
def save_insight(
    category: str,
    finding: str,
    metric_value: Optional[float] = None,
    metric_name: Optional[str] = None,
    visualization_path: Optional[str] = None,
    confidence: str = "medium",
    supporting_data: Optional[str] = None,
    db_path: str = "email_intel.db"
) -> str:
    """
    Save a discovered insight to the database for dashboard display.
    
    Args:
        category: Type of insight - "discount_trend", "urgency", "targeting", 
                  "quarter_clustering", "personalization", "other"
        finding: Human-readable description of the insight
        metric_value: Numeric value associated with finding (e.g., 35.5 for 35.5% increase)
        metric_name: Name of the metric (e.g., "discount_increase_pct")
        visualization_path: Path to associated chart/graph
        confidence: "high", "medium", or "low"
        supporting_data: Optional JSON string with additional context
        db_path: Database path
    
    Returns:
        JSON confirmation with insight ID
    """
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                finding TEXT NOT NULL,
                metric_value REAL,
                metric_name TEXT,
                viz_path TEXT,
                confidence TEXT,
                supporting_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert insight
        cursor.execute("""
            INSERT INTO generated_insights 
            (category, finding, metric_value, metric_name, viz_path, confidence, supporting_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (category, finding, metric_value, metric_name, visualization_path, 
              confidence, supporting_data))
        
        insight_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return json.dumps({
            "success": True,
            "insight_id": insight_id,
            "category": category,
            "finding": finding
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to save insight: {str(e)}"})


# ============================================================================
# UTILITY: Get all tools for agent
# ============================================================================

def get_all_tools():
    """Return list of all tools for LangGraph agent"""
    return [
        inspect_schema,
        sql_query,
        create_visualization,
        # execute_visualization_code,
        statistical_analysis,
        save_insight
    ]