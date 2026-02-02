"""
tools.py
Agent tools for email intelligence analysis
"""

import os
import sqlite3
import pandas as pd
from llm import get_llm
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

from dotenv import load_dotenv
load_dotenv()




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
    viz_description: str,
    cache_key: Optional[str] = None,
    data_query: Optional[str] = None,
    db_path: str = "email_intel.db"
) -> str:
    """
    Generate custom Plotly visualizations using LLM-generated code.
    
    This tool uses an LLM to write Python/Plotly code based on your description,
    then executes it to create beautiful, customized charts.
    
    Args:
        viz_description: Natural language description of the chart you want
                        Example: "Create a line chart showing discount trends over time by brand,
                                 use a dark theme with vibrant colors"
        cache_key: Key to retrieve cached query results (PREFERRED)
        data_query: SQL query to fetch data (only if cache_key not available)
        db_path: Path to database
    
    Returns:
        JSON with visualization path and metadata
    """
    
    try:
        # Get data from cache or query
        if cache_key:
            global _QUERY_CACHE
            if cache_key not in _QUERY_CACHE:
                return json.dumps({
                    "error": f"Cache key '{cache_key}' not found. Run sql_query first.",
                    "available_keys": list(_QUERY_CACHE.keys())
                })
            
            cached_result = _QUERY_CACHE[cache_key]
            df = pd.DataFrame(cached_result["data"])
        
        elif data_query:
            result = json.loads(sql_query.invoke({"query": data_query, "db_path": db_path}))
            
            if "error" in result:
                return json.dumps({"error": result["error"]})
            
            df = pd.DataFrame(result["data"])
        
        else:
            return json.dumps({"error": "Must provide either cache_key or data_query"})
        
        if df.empty:
            return json.dumps({"error": "No data available for visualization"})
        
        # Build prompt for LLM code generation
        code_gen_prompt = f"""Generate Python code using Plotly Express to create this visualization:

USER REQUEST: {viz_description}

AVAILABLE DATA:
- DataFrame variable name: `df`
- Columns: {list(df.columns)}
- Sample data (first 3 rows):
{df.head(3).to_string()}
- Total rows: {len(df)}

REQUIREMENTS:
1. Use plotly.express (imported as `px`)
2. Store the figure in a variable called `fig`
3. Use professional styling with these palettes:
   - Categorical: px.colors.qualitative.Set3 or Vivid or Bold
   - Sequential: px.colors.sequential.Viridis or Plasma or Turbo
   - Diverging: px.colors.diverging.RdYlGn or Spectral
4. Apply these enhancements:
   - template="plotly_white" or "plotly_dark" (choose based on description)
   - height=500
   - hovermode="x unified" or "closest" (choose appropriately)
   - Update axis labels to be human-readable (not database column names)
   - Add title, axis labels, legends as appropriate

5. For specific chart types:
   - Bar charts: Sort by value if not specified otherwise
   - Line charts: Ensure proper date formatting if applicable
   - Heatmaps: Use appropriate color scales
   - Scatter: Consider adding trendlines if comparing relationships

IMPORTANT: Return ONLY executable Python code. 
- No markdown code fences (no ```python)
- No explanations before or after the code
- No comments unless necessary
- Just pure Python code that creates `fig`

Example format (do not copy this, generate fresh code):
df_sorted = df.sort_values('email_count', ascending=False)
fig = px.bar(df_sorted, x='brand', y='email_count', title='Email Frequency by Brand')
fig.update_layout(template='plotly_white', height=500)
"""
        
        # Call LLM to generate code
        llm = get_llm()
        
        # FIXED: Extract content from LLM response
        response = llm.invoke(code_gen_prompt)
        
        if hasattr(response, 'content'):
            generated_code = response.content
        elif isinstance(response, dict) and 'content' in response:
            generated_code = response['content']
        elif isinstance(response, str):
            generated_code = response
        else:
            return json.dumps({
                "error": "Unexpected LLM response format",
                "response_type": str(type(response)),
                "response_preview": str(response)[:200]
        })
        
        # Clean the code (remove markdown fences if LLM added them anyway)
        generated_code = generated_code.strip()
        
        # Remove markdown code fences
        if "```python" in generated_code:
            # Extract code between ```python and ```
            parts = generated_code.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                generated_code = code_part.strip()
        elif "```" in generated_code:
            # Extract code between ``` and ```
            parts = generated_code.split("```")
            if len(parts) >= 3:
                generated_code = parts[1].strip()
            elif len(parts) == 2:
                # Only closing fence
                generated_code = parts[0].strip()
        
        # Additional cleanup
        generated_code = generated_code.strip()
        
        # Validate we have some code
        if not generated_code or len(generated_code) < 10:
            return json.dumps({
                "error": "LLM did not generate valid code",
                "raw_response": str(response)[:500]
            })
        
        # Create execution environment with necessary imports
        exec_globals = {
            'pd': pd,
            'px': px,
            'go': go,
            'make_subplots': make_subplots,
            'df': df,
            'np': np
        }
        
        # Execute the generated code
        try:
            exec(generated_code, exec_globals)
        except Exception as exec_error:
            return json.dumps({
                "error": f"Code execution failed: {str(exec_error)}",
                "generated_code": generated_code,
                "hint": "The LLM generated invalid Python code"
            })
        
        # Get the figure from execution environment
        fig = exec_globals.get('fig')
        
        if fig is None:
            return json.dumps({
                "error": "Generated code did not create a 'fig' variable",
                "generated_code": generated_code,
                "hint": "Code must assign a Plotly figure to variable 'fig'"
            })
        
        # Validate it's a Plotly figure
        if not hasattr(fig, 'write_html'):
            return json.dumps({
                "error": f"'fig' is not a valid Plotly figure (type: {type(fig)})",
                "generated_code": generated_code
            })
        
        # Save to file
        output_dir = Path("output/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"custom_{timestamp}.html"
        filepath = output_dir / filename
        
        fig.write_html(str(filepath))
        
        # Extract chart type from generated code (simple heuristic)
        chart_type = "custom"
        if "px.bar" in generated_code:
            chart_type = "bar"
        elif "px.line" in generated_code:
            chart_type = "line"
        elif "px.scatter" in generated_code:
            chart_type = "scatter"
        elif "px.heatmap" in generated_code or "px.imshow" in generated_code:
            chart_type = "heatmap"
        elif "px.box" in generated_code:
            chart_type = "box"
        elif "px.histogram" in generated_code:
            chart_type = "histogram"
        
        return json.dumps({
            "success": True,
            "viz_path": str(filepath),
            "viz_type": chart_type,
            "description": viz_description,
            "rows_plotted": len(df),
            "columns_used": list(df.columns),
            "used_cache": cache_key is not None,
            "cache_key": cache_key,
            "generated_code": generated_code  # Include for debugging/transparency
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Visualization failed: {str(e)}",
            "generated_code": generated_code if 'generated_code' in locals() else None,
            "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
        })

# ALTERNATIVE: Fallback to predefined templates if code generation fails
@tool
def create_visualization_template(
    viz_type: str,
    title: str,
    x_column: str,
    y_column: str,
    cache_key: Optional[str] = None,
    data_query: Optional[str] = None,
    color_column: Optional[str] = None,
    sort_by: Optional[str] = None,
    color_scheme: str = "Viridis",
    theme: str = "plotly_white",
    db_path: str = "email_intel.db"
) -> str:
    """
    Generate visualizations using predefined templates (fallback method).
    
    Args:
        viz_type: "line", "bar", "scatter", "heatmap", "box", "histogram", "pie"
        title: Chart title
        x_column: Column for x-axis
        y_column: Column for y-axis
        cache_key: Key to retrieve cached data (PREFERRED)
        data_query: SQL query (fallback)
        color_column: Optional column for color grouping
        sort_by: Optional column to sort by
        color_scheme: Plotly color scheme (Viridis, Plasma, Set3, Bold, etc.)
        theme: plotly_white, plotly_dark, simple_white
        db_path: Database path
    """
    
    try:
        # [Keep your existing template-based implementation as fallback]
        # ... (existing code from your current create_visualization function)
        
        # Color scheme mapping
        color_scales = {
            "Viridis": px.colors.sequential.Viridis,
            "Plasma": px.colors.sequential.Plasma,
            "Turbo": px.colors.sequential.Turbo,
            "Set3": px.colors.qualitative.Set3,
            "Bold": px.colors.qualitative.Bold,
            "Vivid": px.colors.qualitative.Vivid,
            "RdYlGn": px.colors.diverging.RdYlGn
        }
        
        # Get data
        if cache_key:
            if cache_key not in _QUERY_CACHE:
                return json.dumps({"error": f"Cache key '{cache_key}' not found"})
            df = pd.DataFrame(_QUERY_CACHE[cache_key]["data"])
        elif data_query:
            result = json.loads(sql_query.invoke({"query": data_query, "db_path": db_path}))
            if "error" in result:
                return json.dumps({"error": result["error"]})
            df = pd.DataFrame(result["data"])
        else:
            return json.dumps({"error": "Must provide cache_key or data_query"})
        
        if df.empty:
            return json.dumps({"error": "No data for visualization"})
        
        # Sort if requested
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        
        # Create visualization
        fig = None
        color_setting = color_scales.get(color_scheme, px.colors.sequential.Viridis)
        
        if viz_type == "bar":
            fig = px.bar(
                df, x=x_column, y=y_column, color=color_column,
                title=title, color_discrete_sequence=color_setting if isinstance(color_setting, list) else None
            )
        elif viz_type == "line":
            fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
        elif viz_type == "scatter":
            fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
        elif viz_type == "pie":
            fig = px.pie(df, names=x_column, values=y_column, title=title)
        # ... add other types
        
        fig.update_layout(template=theme, height=500, hovermode="x unified")
        
        # Save
        output_dir = Path("output/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"{viz_type}_{timestamp}.html"
        fig.write_html(str(filepath))
        
        return json.dumps({
            "success": True,
            "viz_path": str(filepath),
            "viz_type": viz_type,
            "rows_plotted": len(df)
        })
        
    except Exception as e:
        return json.dumps({"error": f"Template visualization failed: {str(e)}"})



# ============================================================================
# TOOL 3: STATISTICAL ANALYSIS (with cache support)
# ============================================================================

def sanitize_for_json(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj



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
    Perform statistical analysis on email data with human-interpretable results.
    
    Args:
        analysis_type: Type of analysis - "trend", "correlation", "outliers", "distribution", "comparison"
        column: Column to analyze
        cache_key: Key to retrieve cached query results (PREFERRED)
        data_query: SQL query to fetch data (only if cache_key not available)
        group_by: Optional column to group analysis by (e.g., by brand)
        db_path: Database path
    
    Returns:
        JSON with statistical results AND natural language interpretations
    """
    
    try:
        # Get data from cache or query
        if cache_key:
            global _QUERY_CACHE
            if cache_key not in _QUERY_CACHE:
                return json.dumps({
                    "error": f"Cache key '{cache_key}' not found. Run sql_query first.",
                    "available_keys": list(_QUERY_CACHE.keys())
                })
            
            cached_result = _QUERY_CACHE[cache_key]
            df = pd.DataFrame(cached_result["data"])
        
        elif data_query:
            result = json.loads(sql_query.invoke({"query": data_query, "db_path": db_path}))
            
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
        
        # TREND ANALYSIS - ENHANCED WITH INTERPRETABILITY
        if analysis_type == "trend":
            if "date" not in df_clean.columns:
                return json.dumps({"error": "Trend analysis requires 'date' column"})
            
            df_clean["date"] = pd.to_datetime(df_clean["date"])
            df_clean = df_clean.sort_values("date")
            
            # Get date range for context
            date_min = df_clean["date"].min()
            date_max = df_clean["date"].max()
            days_span = (date_max - date_min).days
            
            # Convert dates to numeric for regression
            df_clean["date_numeric"] = (df_clean["date"] - date_min).dt.days
            
            if group_by and group_by in df_clean.columns:
                group_results = {}
                
                for group_name, group_df in df_clean.groupby(group_by):
                    if len(group_df) < 5:  # Increased minimum from 3 to 5
                        group_results[str(group_name)] = {
                            "error": f"Insufficient data (only {len(group_df)} points, need 5+)",
                            "interpretation": f"Not enough data to detect meaningful trends for {group_name}"
                        }
                        continue
                    
                    # Get first and last values for period
                    first_val = group_df.iloc[0][column]
                    last_val = group_df.iloc[-1][column]
                    mean_val = group_df[column].mean()
                    
                    # Run regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        group_df["date_numeric"], 
                        group_df[column]
                    )
                    
                    # Calculate absolute change
                    absolute_change = last_val - first_val
                    percent_change = (absolute_change / first_val * 100) if first_val != 0 else 0
                    
                    # Determine if trend is meaningful
                    is_significant = p_value < 0.05
                    is_strong = r_value ** 2 > 0.3  # R² > 0.3 indicates moderate correlation
                    
                    # Build human-readable interpretation
                    if not is_significant:
                        interpretation = (
                            f"{group_name}: No statistically significant trend detected. "
                            f"Values fluctuate around {mean_val:.1f} with no clear direction. "
                            f"(Sample: {len(group_df)} emails over {days_span} days)"
                        )
                    elif not is_strong:
                        interpretation = (
                            f"{group_name}: Weak trend detected but high variability. "
                            f"Changed from {first_val:.1f} to {last_val:.1f} "
                            f"({'+' if absolute_change > 0 else ''}{absolute_change:.1f} points, "
                            f"{'+' if percent_change > 0 else ''}{percent_change:.1f}%) over {days_span} days. "
                            f"However, data is noisy (R²={r_value**2:.2f}). "
                            f"(Sample: {len(group_df)} emails)"
                        )
                    else:
                        # Strong, significant trend
                        direction = "increasing" if slope > 0 else "decreasing"
                        interpretation = (
                            f"{group_name}: Clear {direction} trend. "
                            f"Changed from {first_val:.1f} to {last_val:.1f} "
                            f"({'+' if absolute_change > 0 else ''}{absolute_change:.1f} points, "
                            f"{'+' if percent_change > 0 else ''}{percent_change:.1f}%) over {days_span} days. "
                            f"Strong correlation (R²={r_value**2:.2f}). "
                            f"(Sample: {len(group_df)} emails)"
                        )
                    
                    group_results[str(group_name)] = {
                        "sample_size": int(len(group_df)),
                        "days_analyzed": int(days_span),
                        "first_value": float(first_val),
                        "last_value": float(last_val),
                        "mean_value": float(mean_val),
                        "absolute_change": float(absolute_change),
                        "percent_change": float(percent_change),
                        "slope_per_day": float(slope),
                        "r_squared": float(r_value ** 2),
                        "p_value": float(p_value),
                        "statistically_significant": bool(is_significant),
                        "strong_correlation": bool(is_strong),
                        "interpretation": interpretation
                    }
                
                results = {
                    "trend_by_group": group_results,
                    "date_range": f"{date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}",
                    "total_days": int(days_span)
                }
            
            else:
                # Overall trend (same enhancements)
                first_val = df_clean.iloc[0][column]
                last_val = df_clean.iloc[-1][column]
                mean_val = df_clean[column].mean()
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_clean["date_numeric"], 
                    df_clean[column]
                )
                
                absolute_change = last_val - first_val
                percent_change = (absolute_change / first_val * 100) if first_val != 0 else 0
                
                is_significant = p_value < 0.05
                is_strong = r_value ** 2 > 0.3
                
                if not is_significant:
                    interpretation = (
                        f"No statistically significant trend. Values fluctuate around {mean_val:.1f} "
                        f"with no clear direction over {days_span} days."
                    )
                elif not is_strong:
                    interpretation = (
                        f"Weak trend: changed from {first_val:.1f} to {last_val:.1f} "
                        f"({'+' if absolute_change > 0 else ''}{absolute_change:.1f} points) "
                        f"but high variability (R²={r_value**2:.2f}). "
                        f"Trend may not be reliable."
                    )
                else:
                    direction = "increasing" if slope > 0 else "decreasing"
                    interpretation = (
                        f"Clear {direction} trend: from {first_val:.1f} to {last_val:.1f} "
                        f"({'+' if absolute_change > 0 else ''}{absolute_change:.1f} points, "
                        f"{'+' if percent_change > 0 else ''}{percent_change:.1f}%) "
                        f"over {days_span} days. Strong correlation (R²={r_value**2:.2f})."
                    )
                
                results = {
                    "sample_size": int(len(df_clean)),
                    "days_analyzed": int(days_span),
                    "first_value": float(first_val),
                    "last_value": float(last_val),
                    "mean_value": float(mean_val),
                    "absolute_change": float(absolute_change),
                    "percent_change": float(percent_change),
                    "slope_per_day": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "statistically_significant": bool(is_significant),
                    "strong_correlation": bool(is_strong),
                    "interpretation": interpretation,
                    "date_range": f"{date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}"
                }
        
        # DISTRIBUTION ANALYSIS - ENHANCED
        elif analysis_type == "distribution":
            if group_by and group_by in df_clean.columns:
                group_results = {}
                
                for group_name, group_df in df_clean.groupby(group_by):
                    values = group_df[column].astype(float)
                    
                    if len(values) < 3:
                        group_results[str(group_name)] = {
                            "error": "Insufficient data for distribution analysis"
                        }
                        continue
                    
                    q25 = float(values.quantile(0.25))
                    q75 = float(values.quantile(0.75))
                    iqr = q75 - q25
                    
                    group_results[str(group_name)] = {
                        "count": int(len(values)),
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "q25": q25,
                        "q75": q75,
                        "iqr": iqr,
                        "range": float(values.max() - values.min()),
                        "interpretation": (
                            f"{group_name}: Average {values.mean():.1f}, ranging from "
                            f"{values.min():.1f} to {values.max():.1f}. "
                            f"Middle 50% fall between {q25:.1f} and {q75:.1f}. "
                            f"({'High' if values.std() / values.mean() > 0.3 else 'Low'} variability)"
                        )
                    }
                
                results = {"distribution_by_group": group_results}
            
            else:
                values = df_clean[column].astype(float)
                q25 = float(values.quantile(0.25))
                q75 = float(values.quantile(0.75))
                iqr = q75 - q25
                
                results = {
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr,
                    "range": float(values.max() - values.min()),
                    "interpretation": (
                        f"Values average {values.mean():.1f}, ranging from {values.min():.1f} "
                        f"to {values.max():.1f}. Half the values are between {q25:.1f} and {q75:.1f}. "
                        f"Variability is {'high' if values.std() / values.mean() > 0.3 else 'moderate'}."
                    )
                }
        
        # COMPARISON ANALYSIS - NEW
        elif analysis_type == "comparison":
            if not group_by or group_by not in df_clean.columns:
                return json.dumps({"error": "Comparison analysis requires group_by parameter"})
            
            group_stats = df_clean.groupby(group_by)[column].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            
            # Rank groups by mean
            ranked = group_stats.sort_values('mean', ascending=False)
            
            comparison_results = {}
            for idx, (group_name, row) in enumerate(ranked.iterrows(), 1):
                comparison_results[str(group_name)] = {
                    "rank": idx,
                    "mean": float(row['mean']),
                    "median": float(row['median']),
                    "count": int(row['count']),
                    "range": f"{row['min']:.1f} - {row['max']:.1f}"
                }
            
            # Overall interpretation
            top_group = ranked.index[0]
            top_value = ranked.iloc[0]['mean']
            bottom_group = ranked.index[-1]
            bottom_value = ranked.iloc[-1]['mean']
            
            results = {
                "comparison_by_group": comparison_results,
                "interpretation": (
                    f"{top_group} has the highest average {column} at {top_value:.1f}, "
                    f"while {bottom_group} has the lowest at {bottom_value:.1f}. "
                    f"That's a {((top_value - bottom_value) / bottom_value * 100):.1f}% difference."
                )
            }
        
        # OUTLIER DETECTION - ENHANCED
        elif analysis_type == "outliers":
            values = df_clean[column].astype(float)
            mean_val = values.mean()
            std_val = values.std()
            z_scores = np.abs(stats.zscore(values))
            outliers = df_clean[z_scores > 2.5].copy()  # Slightly relaxed threshold
            outliers['z_score'] = z_scores[z_scores > 2.5]
            
            results = {
                "outliers_found": len(outliers),
                "outlier_threshold": "z-score > 2.5 (unusual values)",
                "mean": float(mean_val),
                "std": float(std_val),
                "outliers": outliers[[column, 'z_score']].head(10).to_dict('records'),
                "interpretation": (
                    f"Found {len(outliers)} unusual values out of {len(values)} total. "
                    f"Normal range is approximately {mean_val - 2*std_val:.1f} to {mean_val + 2*std_val:.1f}."
                ) if len(outliers) > 0 else "No significant outliers detected."
            }
        
        else:
            return json.dumps({
                "error": f"Unknown analysis_type: {analysis_type}",
                "supported": ["trend", "distribution", "comparison", "outliers"]
            })
        
        safe_results = sanitize_for_json(results)

        return json.dumps({
            "success": True,
            "analysis_type": analysis_type,
            "column": column,
            "results": safe_results,
            "used_cache": cache_key is not None,
            "cache_key": cache_key,
            "⚠️_important": "Use the 'interpretation' field for conversational responses"
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

def get_all_tools(mode: str = "chat"):
    """
    Return list of all tools for LangGraph agent
    
    Args:
        mode: "chat" or "analysis"
            - chat: excludes save_insight (for conversations)
            - analysis: includes all tools (for scheduled reports)
    """
    base_tools = [
        inspect_schema,
        sql_query,
        create_visualization,
        statistical_analysis,
    ]
    
    # Only include save_insight in analysis mode
    if mode == "analysis":
        base_tools.append(save_insight)
    
    return base_tools