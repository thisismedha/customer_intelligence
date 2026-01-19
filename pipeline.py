"""
pipeline.py
Complete pipeline: Email extraction → Schema parsing → ReAct Agent Analysis
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()

from agent import EmailIntelligenceAgent, print_agent_conversation


API_KEY = os.getenv("GOOGLE_API_KEY")



# ============================================================================
# STEP 1: DATABASE INITIALIZATION
# ============================================================================

def init_database(db_path: str = "email_intel.db"):
    """Initialize SQLite database with schema"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Main emails table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            brand TEXT NOT NULL,
            subject TEXT,
            date TEXT,
            discount_percent REAL,
            discount_type TEXT,
            urgency_level TEXT,
            urgency_phrases TEXT,
            targeting_signal TEXT,
            quarter_end_proximity BOOLEAN,
            personalized BOOLEAN,
            raw_snippet TEXT,
            raw_body TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Brand insights table (aggregated stats)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS brand_insights (
            brand TEXT PRIMARY KEY,
            total_emails INTEGER,
            avg_discount REAL,
            urgency_frequency REAL,
            reactivation_rate REAL,
            quarter_clustering_score REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Generated insights table (from agent)
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
    
    conn.commit()
    return conn


# ============================================================================
# STEP 2: SCHEMA EXTRACTION
# ============================================================================

EXTRACTION_SCHEMA = """
{
  "brand": "string (extract from 'from' field, just company name)",
  "subject": "string",
  "date": "ISO 8601 string",
  "discount_percent": "number | null (main discount %, e.g., 50 for '50% off')",
  "discount_type": "percentage | flat_amount | tiered | free_shipping | none",
  "urgency_level": "high | medium | low | none",
  "urgency_phrases": ["list of urgency terms found"],
  "targeting_signal": "reactivation | new_customer | loyalty | generic",
  "quarter_end_proximity": "boolean (true if within 7 days of quarter end)",
  "personalized": "boolean (uses recipient name or 'you' language)"
}
"""

def extract_email_data(email: Dict, llm: ChatGoogleGenerativeAI) -> Dict:
    """Extract structured data from a single email"""
    
    EXTRACTION_PROMPT = f"""Extract promotional email data into this JSON schema:

{EXTRACTION_SCHEMA}

Email:
From: {email.get('from', '')}
Subject: {email.get('subject', '')}
Date: {email.get('date', '')}
Snippet: {email.get('snippet', '')[:500]}

Rules:
1. brand: Extract just company name (e.g., "Banana Republic" from "Banana Republic Factory <email@...>")
2. discount_percent: Primary discount number (e.g., "50% off" → 50, "Buy 1 Get 1" → null)
3. urgency_level:
   - high: "today only", "ends tonight", "last chance", "hours left"
   - medium: "this weekend", "limited time", "while supplies last"
   - low: "don't miss", "hurry"
   - none: no urgency
4. targeting_signal:
   - reactivation: "we miss you", "come back", "it's been a while"
   - new_customer: "welcome", "first order", "new member"
   - loyalty: "VIP", "exclusive for you", "member only"
   - generic: standard promotion
5. quarter_end_proximity: true if date is Mar 25-31, Jun 25-30, Sep 25-30, or Dec 25-31

Return ONLY valid JSON, no markdown, no explanation.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=EXTRACTION_PROMPT)])
        
        # Clean response
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        data = json.loads(content)
        data["id"] = email.get("id")
        return data
        
    except Exception as e:
        print(f"⚠️  Failed to extract {email.get('id')}: {e}")
        return None

def get_processed_email_ids(cursor) -> set[str]:
    cursor.execute("SELECT id FROM emails")
    return {row[0] for row in cursor.fetchall()}


def extract_all_emails(
    emails: List[Dict], 
    db_path: str = "email_intel.db",
    batch_size: int = 50
) -> int:
    """
    Extract structured data from all emails and save to database.
    
    Args:
        emails: List of raw email dicts
        db_path: Database path
        batch_size: Commit to DB every N emails
        
    Returns:
        Number of successfully parsed emails
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=API_KEY,
        temperature=0
    )
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    parsed_count = 0
    skipped_count = 0

    processed_ids = get_processed_email_ids(cursor)
    for i, email in enumerate(emails, 1):
        email_id = email.get("id")

        if parsed_count==1:
            break

        if email_id in processed_ids:
            skipped_count += 1
            continue
        data = extract_email_data(email, llm)
        
        if data:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO emails 
                    (id, brand, subject, date, discount_percent, discount_type,
                     urgency_level, urgency_phrases, targeting_signal,
                     quarter_end_proximity, personalized, raw_snippet, raw_body)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["id"],
                    data["brand"],
                    data["subject"],
                    data["date"],
                    data.get("discount_percent"),
                    data["discount_type"],
                    data["urgency_level"],
                    json.dumps(data.get("urgency_phrases", [])),
                    data["targeting_signal"],
                    data["quarter_end_proximity"],
                    data["personalized"],
                    email.get("snippet", ""),
                    email.get("body", "")
                ))
                processed_ids.add(email_id)
                parsed_count += 1
                
                if parsed_count % batch_size == 0:
                    conn.commit()
                    print(f"✓ Parsed {parsed_count}/{len(emails)} emails...")
                    print(f"✓ Skipped {skipped_count}/{len(emails)} emails...")

                    
            except Exception as e:
                print(f"⚠️  Database error for {data['id']}: {e}")
    
    conn.commit()
    conn.close()
    
    return parsed_count


# ============================================================================
# STEP 3: AGGREGATE BRAND STATISTICS
# ============================================================================

def aggregate_brand_stats(db_path: str = "email_intel.db"):
    """Calculate and cache per-brand statistics"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    brands = cursor.execute("SELECT DISTINCT brand FROM emails").fetchall()
    
    for (brand,) in brands:
        stats = cursor.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(discount_percent) as avg_discount,
                SUM(CASE WHEN urgency_level IN ('high', 'medium') THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as urgency_freq,
                SUM(CASE WHEN targeting_signal = 'reactivation' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as reactivation_rate,
                SUM(CASE WHEN quarter_end_proximity = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as quarter_score
            FROM emails
            WHERE brand = ?
        """, (brand,)).fetchone()
        
        cursor.execute("""
            INSERT OR REPLACE INTO brand_insights
            (brand, total_emails, avg_discount, urgency_frequency,
             reactivation_rate, quarter_clustering_score, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, (brand, stats[0], stats[1], stats[2], stats[3], stats[4]))
    
    conn.commit()
    conn.close()
    
    print(f"✓ Aggregated stats for {len(brands)} brands")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    emails_file: str = "emails.json",
    db_path: str = "email_intel.db",
    run_agent: bool = True
):
    """
    Run the complete email intelligence pipeline.
    
    Steps:
        1. Load raw emails from JSON
        2. Initialize database
        3. Extract structured data using LLM
        4. Aggregate brand statistics
        5. Run ReAct agent for autonomous analysis
    """
    
    print("="*80)
    print("📧 EMAIL COMPETITIVE INTELLIGENCE PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Load emails
    print("📂 Loading emails...")
    with open(emails_file, "r") as f:
        raw_emails = json.load(f)
    print(f"✓ Loaded {len(raw_emails)} emails\n")
    
    # Step 2: Initialize database
    print("🗄️  Initializing database...")
    init_database(db_path)
    print(f"✓ Database ready: {db_path}\n")
    
    # Step 3: Extract structured data
    print("🔍 Extracting structured data with LLM...")
    parsed_count = extract_all_emails(raw_emails, db_path)
    print(f"✓ Successfully parsed {parsed_count}/{len(raw_emails)} emails\n")
    
    # Step 4: Aggregate statistics
    print("📊 Aggregating brand statistics...")
    aggregate_brand_stats(db_path)
    print()
    
    # Step 5: Run ReAct agent
    if run_agent:
        print("🤖 Launching ReAct Agent for autonomous analysis...")
        print("-" * 80)
        
        agent = EmailIntelligenceAgent()
        result = agent.analyze_automatically()
        
        print("\n" + "="*80)
        print("📈 ANALYSIS COMPLETE")
        print("="*80)
        print(f"\n✅ Insights generated: {result['insights_generated']}")
        print(f"📊 Visualizations created: {result['visualizations_created']}")
        print(f"🔄 Total reasoning steps: {result['total_steps']}")
        
        if result.get("final_summary"):
            print(f"\n📝 Summary:\n{result['final_summary'][:500]}...\n")
        
        # Print reasoning trace
        print_agent_conversation(result["full_conversation"])
        
        # Show where to find outputs
        print("="*80)
        print("📁 OUTPUT LOCATIONS")
        print("="*80)
        print(f"Database: {db_path}")
        print(f"Visualizations: output/visualizations/")
        print(f"Insights table: generated_insights (in database)")
        print("\n💡 Next: View insights in dashboard or query the database!")
        print("="*80 + "\n")


# ============================================================================
# EXAMPLE: CONVERSATIONAL USAGE
# ============================================================================

def example_conversation():
    """Example of using agent conversationally"""
    
    agent = EmailIntelligenceAgent()
    
    questions = [
        "Which brand has the highest average discount?",
        "Show me discount trends for Banana Republic over time",
        "Which brands use fake urgency language?"
    ]
    
    for question in questions:
        print(f"\n❓ {question}")
        result = agent.answer_question(question)
        print(f"✅ {result['answer'][:200]}...")
        if result['visualizations']:
            print(f"📊 Created: {result['visualizations']}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run full pipeline
    run_pipeline(
        emails_file="promotional_emails.json",
        db_path="email_intel.db",
        run_agent=True
    )
    
    # Optionally: Run conversational examples
    # example_conversation()