"""
pipeline.py
Enhanced Email Intelligence Pipeline - Phase 1 with Engagement Tracking
"""

import sqlite3
import json
import os
from typing import Dict, List
from datetime import datetime
from agent import EmailIntelligenceAgent
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================================
# STEP 1: ENHANCED DATABASE SCHEMA
# ============================================================================

def init_database(db_path: str = "email_intel.db"):
    """Initialize SQLite database with enhanced schema"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enhanced emails table with Phase 1 fields
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            brand TEXT NOT NULL,
            subject TEXT,
            date TEXT,
            
            -- Engagement Tracking (NEW!)
            is_read BOOLEAN DEFAULT 0,
            read_timestamp TEXT,
            engagement_score REAL,
            
            -- Enhanced Pricing (Phase 1)
            discount_percent REAL,
            discount_type TEXT,
            price_anchor TEXT,
            minimum_purchase REAL,
            price_positioning TEXT,
            
            -- Product Intelligence (Phase 1)
            product_categories TEXT,
            product_specificity TEXT,
            
            -- Temporal Strategy (Phase 1)
            promotion_duration TEXT,
            day_of_week INTEGER,
            time_of_day TEXT,
            is_holiday_adjacent BOOLEAN,
            season TEXT,
            
            -- Urgency & Scarcity
            urgency_level TEXT,
            urgency_phrases TEXT,
            inventory_scarcity BOOLEAN,
            countdown_present BOOLEAN,
            
            -- Targeting & Lifecycle (Phase 1)
            targeting_signal TEXT,
            lifecycle_stage TEXT,
            personalized BOOLEAN,
            
            -- Metadata
            quarter_end_proximity BOOLEAN,
            raw_snippet TEXT,
            raw_body TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Enhanced brand insights with engagement metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS brand_insights (
            brand TEXT PRIMARY KEY,
            total_emails INTEGER,
            
            -- Engagement Metrics (NEW!)
            read_rate REAL,
            avg_engagement_score REAL,
            unread_count INTEGER,
            
            -- Pricing Strategy
            avg_discount REAL,
            uses_anchor_pricing_pct REAL,
            price_positioning_mode TEXT,
            
            -- Product Strategy
            product_specificity_score REAL,
            top_categories TEXT,
            
            -- Temporal Patterns
            best_send_day INTEGER,
            best_send_time TEXT,
            holiday_clustering_pct REAL,
            
            -- Lifecycle
            urgency_frequency REAL,
            reactivation_rate REAL,
            vip_targeting_pct REAL,
            
            -- Metadata
            quarter_clustering_score REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Generated insights table (unchanged)
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
    
    # Create indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_brand ON emails(brand)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_date ON emails(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_read ON emails(is_read)")
    
    conn.commit()
    return conn


# ============================================================================
# STEP 2: ENHANCED EXTRACTION SCHEMA
# ============================================================================

EXTRACTION_SCHEMA = """
{
  "brand": "string (extract from 'from' field, just company name)",
  "subject": "string",
  "date": "ISO 8601 string",
  
  "discount_percent": "number | null (main discount %, e.g., 50 for '50% off')",
  "discount_type": "percentage | flat_amount | tiered | free_shipping | bundle | none",
  "price_anchor": "string | null (e.g., 'was $99' or 'originally $149')",
  "minimum_purchase": "number | null (e.g., 50 for 'free shipping over $50')",
  "price_positioning": "premium | mid-tier | value",
  
  "product_categories": ["array of product categories mentioned, e.g., ['blouses', 'button-downs']"],
  "product_specificity": "generic | category | specific_item",
  
  "promotion_duration": "flash | daily | weekend | week_long | ongoing",
  "day_of_week": "integer (0=Monday, 6=Sunday)",
  "time_of_day": "morning | afternoon | evening | night",
  "is_holiday_adjacent": "boolean (within 3 days of major US holiday)",
  "season": "spring | summer | fall | winter | holiday",
  
  "urgency_level": "high | medium | low | none",
  "urgency_phrases": ["list of urgency terms found"],
  "inventory_scarcity": "boolean (mentions 'low stock', 'almost gone', 'selling fast')",
  "countdown_present": "boolean (has explicit countdown timer or 'X hours left')",
  
  "targeting_signal": "acquisition | activation | retention | winback | vip | generic",
  "lifecycle_stage": "new | occasional | regular | vip",
  "personalized": "boolean (uses recipient name or 'you' language)"
}
"""


def extract_email_data(email: Dict, llm: ChatGoogleGenerativeAI) -> Dict:
    """Extract structured data from a single email with Phase 1 enhancements"""
    
    EXTRACTION_PROMPT = f"""Extract promotional email data into this JSON schema:

{EXTRACTION_SCHEMA}

Email:
From: {email.get('from', '')}
Subject: {email.get('subject', '')}
Date: {email.get('date', '')}
Snippet: {email.get('snippet', '')[:500]}

EXTRACTION RULES:

**Brand:**
- Extract just company name (e.g., "Banana Republic" from "Banana Republic Factory <email@...>")

**Pricing:**
- discount_percent: Primary discount number (e.g., "50% off" → 50, "Buy 1 Get 1" → null)
- price_anchor: Look for "was $X", "originally $Y", "regular price $Z"
- minimum_purchase: Extract threshold (e.g., "$50 minimum" → 50, "free shipping over $75" → 75)
- price_positioning:
  * premium: luxury language, high-end brands, "craftsmanship", "quality materials"
  * mid-tier: balanced value/quality
  * value: emphasis on savings, "affordable", "budget-friendly"

**Product Intelligence:**
- product_categories: Extract all mentioned categories (e.g., ["blouses", "button-downs", "pants"])
- product_specificity:
  * generic: "everything", "sitewide", "all items"
  * category: specific categories but not individual items
  * specific_item: exact products like "cashmere sweater", "leather jacket"

**Temporal:**
- promotion_duration:
  * flash: "today only", "next 2 hours"
  * daily: "24 hours", "one day"
  * weekend: "this weekend", "Saturday & Sunday"
  * week_long: "all week", "7 days"
  * ongoing: no end time mentioned
- day_of_week: 0=Monday, 1=Tuesday, ..., 6=Sunday (based on date field)
- time_of_day: morning (6am-12pm), afternoon (12pm-5pm), evening (5pm-9pm), night (9pm-6am)
- is_holiday_adjacent: true if within 3 days of Christmas, Thanksgiving, New Year, Easter, Memorial Day, Labor Day, 4th of July, Valentine's, Mother's/Father's Day
- season: Based on date - spring (Mar-May), summer (Jun-Aug), fall (Sep-Nov), winter (Dec-Feb), holiday (Nov 15-Jan 5)

**Urgency:**
- urgency_level:
  * high: "today only", "ends tonight", "last chance", "hours left", "ending soon"
  * medium: "this weekend", "limited time", "while supplies last", "don't miss"
  * low: "hurry", "act now"
  * none: no urgency language
- urgency_phrases: Extract all urgency-related phrases found
- inventory_scarcity: Look for "low stock", "almost gone", "selling fast", "limited quantities"
- countdown_present: true if mentions specific time countdown like "6 hours left", "ends at midnight"

**Lifecycle Targeting:**
- targeting_signal:
  * acquisition: "new customer", "first order", "welcome"
  * activation: "complete your profile", "verify email"
  * retention: "thank you", "valued customer"
  * winback: "we miss you", "come back", "it's been a while"
  * vip: "VIP", "exclusive", "member only", "platinum"
  * generic: standard promotion
- lifecycle_stage:
  * new: acquisition messaging
  * occasional: generic promotions
  * regular: loyalty/retention focus
  * vip: exclusive/member language
- personalized: true if uses recipient name or heavy use of "you", "your"

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
        
        # Calculate engagement from labels
        labels = email.get("labels", [])
        data["is_read"] = "UNREAD" not in labels
        
        # Calculate engagement score (0-100)
        # Read = 50 points, plus bonus for other engagement signals
        engagement = 50 if data["is_read"] else 0
        # Bonus points (to be enhanced later with actual engagement data)
        # For now, we'll set baseline
        data["engagement_score"] = engagement
        
        return data
        
    except Exception as e:
        print(f"⚠️  Failed to extract {email.get('id')}: {e}")
        return None


def get_processed_email_ids(cursor) -> set:
    """Get set of already processed email IDs"""
    cursor.execute("SELECT id FROM emails")
    return {row[0] for row in cursor.fetchall()}


def extract_all_emails(
    emails: List[Dict], 
    db_path: str = "email_intel.db",
    batch_size: int = 50,
    max_emails: int = None
) -> int:
    """
    Extract structured data from all emails and save to database.
    
    Args:
        emails: List of raw email dicts
        db_path: Database path
        batch_size: Commit to DB every N emails
        max_emails: Maximum emails to process (None for all)
        
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
        
        # Limit processing if max_emails specified
        if max_emails and parsed_count >= max_emails:
            print(f"✓ Reached max_emails limit ({max_emails})")
            break
        
        # Skip already processed
        if email_id in processed_ids:
            skipped_count += 1
            continue
        
        # Extract data
        data = extract_email_data(email, llm)
        
        if data:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO emails 
                    (id, brand, subject, date, 
                     is_read, engagement_score,
                     discount_percent, discount_type, price_anchor, minimum_purchase, price_positioning,
                     product_categories, product_specificity,
                     promotion_duration, day_of_week, time_of_day, is_holiday_adjacent, season,
                     urgency_level, urgency_phrases, inventory_scarcity, countdown_present,
                     targeting_signal, lifecycle_stage, personalized,
                     quarter_end_proximity, raw_snippet, raw_body)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["id"],
                    data["brand"],
                    data["subject"],
                    data["date"],
                    data["is_read"],
                    data["engagement_score"],
                    data.get("discount_percent"),
                    data["discount_type"],
                    data.get("price_anchor"),
                    data.get("minimum_purchase"),
                    data["price_positioning"],
                    json.dumps(data.get("product_categories", [])),
                    data["product_specificity"],
                    data["promotion_duration"],
                    data["day_of_week"],
                    data["time_of_day"],
                    data["is_holiday_adjacent"],
                    data["season"],
                    data["urgency_level"],
                    json.dumps(data.get("urgency_phrases", [])),
                    data.get("inventory_scarcity", False),
                    data.get("countdown_present", False),
                    data["targeting_signal"],
                    data["lifecycle_stage"],
                    data["personalized"],
                    False,  # quarter_end_proximity - calculate if needed
                    email.get("snippet", ""),
                    email.get("body", "")
                ))
                
                processed_ids.add(email_id)
                parsed_count += 1
                
                if parsed_count % batch_size == 0:
                    conn.commit()
                    print(f"✓ Parsed {parsed_count}/{len(emails)} emails...")
                    print(f"✓ Skipped {skipped_count} already processed emails...")
                    
            except Exception as e:
                print(f"⚠️  Database error for {data['id']}: {e}")
    
    conn.commit()
    conn.close()
    
    return parsed_count


# ============================================================================
# STEP 3: ENHANCED BRAND STATISTICS
# ============================================================================

def aggregate_brand_stats(db_path: str = "email_intel.db"):
    """Calculate and cache per-brand statistics with Phase 1 enhancements"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    brands = cursor.execute("SELECT DISTINCT brand FROM emails").fetchall()
    
    for (brand,) in brands:
        # Get comprehensive stats (without MODE function)
        stats = cursor.execute("""
            SELECT 
                COUNT(*) as total,
                
                -- Engagement Metrics
                SUM(CASE WHEN is_read = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as read_rate,
                AVG(engagement_score) as avg_engagement,
                SUM(CASE WHEN is_read = 0 THEN 1 ELSE 0 END) as unread_count,
                
                -- Pricing Strategy
                AVG(discount_percent) as avg_discount,
                SUM(CASE WHEN price_anchor IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as anchor_pct,
                
                -- Product Strategy (avg categories per email)
                AVG(LENGTH(product_categories) - LENGTH(REPLACE(product_categories, ',', '')) + 1) as cat_count,
                
                -- Temporal Patterns (holiday clustering only, day/time calculated separately)
                SUM(CASE WHEN is_holiday_adjacent = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as holiday_pct,
                
                -- Lifecycle
                SUM(CASE WHEN urgency_level IN ('high', 'medium') THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as urgency_freq,
                SUM(CASE WHEN targeting_signal = 'winback' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as reactivation_rate,
                SUM(CASE WHEN lifecycle_stage = 'vip' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as vip_pct
                
            FROM emails
            WHERE brand = ?
        """, (brand,)).fetchone()
        
        # Get most common day of week (MODE replacement)
        best_day_result = cursor.execute("""
            SELECT day_of_week, COUNT(*) as cnt
            FROM emails
            WHERE brand = ? AND day_of_week IS NOT NULL
            GROUP BY day_of_week
            ORDER BY cnt DESC
            LIMIT 1
        """, (brand,)).fetchone()
        
        best_day = best_day_result[0] if best_day_result else None
        
        # Get most common time of day (MODE replacement)
        best_time_result = cursor.execute("""
            SELECT time_of_day, COUNT(*) as cnt
            FROM emails
            WHERE brand = ? AND time_of_day IS NOT NULL
            GROUP BY time_of_day
            ORDER BY cnt DESC
            LIMIT 1
        """, (brand,)).fetchone()
        
        best_time = best_time_result[0] if best_time_result else None
        
        # Get most common price positioning
        price_pos = cursor.execute("""
            SELECT price_positioning, COUNT(*) as cnt
            FROM emails
            WHERE brand = ? AND price_positioning IS NOT NULL
            GROUP BY price_positioning
            ORDER BY cnt DESC
            LIMIT 1
        """, (brand,)).fetchone()
        
        price_positioning_mode = price_pos[0] if price_pos else None
        
        # Get top 3 product categories
        top_cats = cursor.execute("""
            WITH categories AS (
                SELECT TRIM(value) as category
                FROM emails, json_each(product_categories)
                WHERE brand = ?
            )
            SELECT category, COUNT(*) as cnt
            FROM categories
            WHERE category != ''
            GROUP BY category
            ORDER BY cnt DESC
            LIMIT 3
        """, (brand,)).fetchall()
        
        top_categories = json.dumps([cat[0] for cat in top_cats])
        
        # Insert aggregated stats
        cursor.execute("""
            INSERT OR REPLACE INTO brand_insights
            (brand, total_emails, 
             read_rate, avg_engagement_score, unread_count,
             avg_discount, uses_anchor_pricing_pct, price_positioning_mode,
             product_specificity_score, top_categories,
             best_send_day, best_send_time, holiday_clustering_pct,
             urgency_frequency, reactivation_rate, vip_targeting_pct,
             last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            brand,
            stats[0],  # total
            stats[1],  # read_rate
            stats[2],  # avg_engagement
            stats[3],  # unread_count
            stats[4],  # avg_discount
            stats[5],  # anchor_pct
            price_positioning_mode,
            stats[6],  # cat_count
            top_categories,
            best_day,  # best_day (calculated separately)
            best_time, # best_time (calculated separately)
            stats[7],  # holiday_pct (was index 9, now 7)
            stats[8],  # urgency_freq (was index 10, now 8)
            stats[9],  # reactivation_rate (was index 11, now 9)
            stats[10]  # vip_pct (was index 12, now 10)
        ))
    
    conn.commit()
    conn.close()
    
    print(f"✓ Aggregated enhanced stats for {len(brands)} brands")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    emails_file: str = "emails.json",
    db_path: str = "email_intel.db",
    max_emails: int = None,
    run_agent: bool = False
):
    """
    Run the enhanced email intelligence pipeline.
    
    Steps:
        1. Load raw emails from JSON
        2. Initialize enhanced database
        3. Extract structured data with Phase 1 fields
        4. Aggregate brand statistics with engagement metrics
        5. Optionally run ReAct agent for autonomous analysis
    
    Args:
        emails_file: Path to JSON file with raw emails
        db_path: SQLite database path
        max_emails: Max emails to process (None for all)
        run_agent: Whether to run autonomous analysis
    """
    
    print("="*80)
    print("📧 ENHANCED EMAIL COMPETITIVE INTELLIGENCE PIPELINE - PHASE 1")
    print("="*80 + "\n")
    
    # Step 1: Load emails
    print("📂 Loading emails...")
    with open(emails_file, "r") as f:
        raw_emails = json.load(f)
    print(f"✓ Loaded {len(raw_emails)} emails\n")
    
    # Step 2: Initialize database
    print("🗄️  Initializing enhanced database...")
    init_database(db_path)
    print(f"✓ Database ready with Phase 1 schema: {db_path}\n")
    
    # Step 3: Extract structured data
    print("🔍 Extracting enhanced data with LLM...")
    print("   Phase 1 enhancements:")
    print("   ✓ Engagement tracking (read/unread)")
    print("   ✓ Price positioning & anchoring")
    print("   ✓ Product categories & specificity")
    print("   ✓ Temporal strategy (day/time/season)")
    print("   ✓ Lifecycle stage detection\n")
    
    parsed_count = extract_all_emails(raw_emails, db_path, max_emails=max_emails)
    print(f"✓ Successfully parsed {parsed_count}/{len(raw_emails)} emails\n")
    
    # Step 4: Aggregate statistics
    print("📊 Aggregating enhanced brand statistics...")
    aggregate_brand_stats(db_path)
    print()
    
    # Step 5: Show sample insights
    print("="*80)
    print("📈 SAMPLE ENGAGEMENT INSIGHTS")
    print("="*80 + "\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    engagement_stats = cursor.execute("""
        SELECT 
            brand,
            total_emails,
            ROUND(read_rate, 1) as read_rate,
            unread_count,
            best_send_day,
            best_send_time,
            ROUND(avg_discount, 1) as avg_disc
        FROM brand_insights
        ORDER BY read_rate DESC
        LIMIT 5
    """).fetchall()
    
    print("Top Brands by Engagement (Read Rate):")
    print("-" * 80)
    for stat in engagement_stats:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day = day_names[stat[4]] if stat[4] is not None else "N/A"
        print(f"  {stat[0]:<25} | Read Rate: {stat[2]}% | Unread: {stat[3]} | Best: {day} {stat[5]} | Avg Disc: {stat[6]}%")
    
    conn.close()
    print()
    
    # Step 6: Run agent if requested
    if run_agent:
        print("🤖 Launching ReAct Agent for autonomous analysis...")
        print("-" * 80)
        
        from agent import EmailIntelligenceAgent
        
        agent = EmailIntelligenceAgent(mode="analysis")
        result = agent.analyze_automatically()
        
        print("\n" + "="*80)
        print("📈 ANALYSIS COMPLETE")
        print("="*80)
        print(f"\n✅ Insights generated: {result['insights_generated']}")
        print(f"📊 Visualizations created: {result['visualizations_created']}")
        print(f"🔄 Total reasoning steps: {result['total_steps']}")
        
        if result.get("final_summary"):
            print(f"\n📝 Summary:\n{result['final_summary'][:500]}...\n")
    
    print("="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)


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
    
