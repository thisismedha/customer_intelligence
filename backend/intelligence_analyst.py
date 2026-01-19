"""
SQL-Powered Intelligence Analyst Agent
Converts JSON to SQLite for accurate querying, then uses LLM for insights
"""
import json
import sqlite3
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd

class SQLIntelligenceAnalyst:
    """Agent that queries structured data accurately, then analyzes with LLM"""
    
    def __init__(self, intelligence_file: str = "email_intelligence.json"):
        """Initialize analyst with SQL backend"""
        
        print("🤖 Initializing SQL-Powered Intelligence Analyst...")
        
        # Load JSON data
        with open(intelligence_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize SQLite database
        self.db_path = "email_intelligence.db"
        self._create_database()
        
        # Initialize LLM (only for interpretation, not data retrieval)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        print(f"✓ Loaded {self._count_emails()} emails into SQLite")
        print(f"✓ Tracking {self._count_brands()} brands\n")
    
    def _create_database(self):
        """Create SQLite database from JSON"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create emails table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                email_id TEXT PRIMARY KEY,
                brand_name TEXT,
                brand_domain TEXT,
                subject TEXT,
                received_date TEXT,
                discount_value REAL,
                has_discount INTEGER,
                stacked_discounts INTEGER,
                free_shipping INTEGER,
                coupon_code TEXT,
                urgency_score REAL,
                exclusivity_score REAL,
                personalization_score REAL,
                has_urgency INTEGER,
                urgency_type TEXT,
                flash_sale INTEGER,
                countdown_timer INTEGER,
                hour INTEGER,
                day_of_week TEXT,
                day_of_month INTEGER,
                month INTEGER,
                quarter TEXT,
                is_weekend INTEGER,
                is_month_end INTEGER,
                is_quarter_end INTEGER,
                is_holiday_period INTEGER,
                personalized INTEGER,
                cart_abandonment INTEGER,
                winback_campaign INTEGER,
                email_length INTEGER
            )
        """)
        
        # Create brand_stats table for aggregated metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS brand_stats (
                brand_name TEXT PRIMARY KEY,
                total_emails INTEGER,
                first_email_date TEXT,
                last_email_date TEXT,
                avg_discount REAL,
                max_discount REAL,
                min_discount REAL,
                discount_frequency REAL,
                avg_urgency_score REAL,
                month_end_frequency REAL,
                quarter_end_frequency REAL,
                personalization_frequency REAL
            )
        """)
        
        # Insert email data
        for email in self.data.get('processed_emails', []):
            cursor.execute("""
                INSERT OR REPLACE INTO emails VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                email['email_id'],
                email['brand_info']['name'],
                email['brand_info']['domain'],
                email['raw_metadata']['subject'],
                email['raw_metadata']['received_date'],
                email['pricing_signals'].get('primary_discount_value'),
                int(email['pricing_signals']['has_discount']),
                int(email['pricing_signals'].get('stacked_discounts', False)),
                int(email['pricing_signals']['free_shipping']),
                email['pricing_signals'].get('coupon_code'),
                email['language_signals']['urgency_score'],
                email['language_signals']['exclusivity_score'],
                email['language_signals']['personalization_score'],
                int(email['timing_signals']['has_urgency']),
                email['timing_signals'].get('urgency_type'),
                int(email['timing_signals'].get('flash_sale', False)),
                int(email['timing_signals'].get('countdown_timer', False)),
                email['timing_signals']['send_datetime']['hour'],
                email['timing_signals']['send_datetime']['day_of_week'],
                email['timing_signals']['send_datetime']['day_of_month'],
                email['timing_signals']['send_datetime']['month'],
                email['timing_signals']['send_datetime']['quarter'],
                int(email['timing_signals']['send_datetime']['is_weekend']),
                int(email['timing_signals']['send_datetime']['is_month_end']),
                int(email['timing_signals']['send_datetime']['is_quarter_end']),
                int(email['timing_signals']['send_datetime']['is_holiday_period']),
                int(email['targeting_signals']['personalized']),
                int(email['targeting_signals']['behavioral_triggers']['cart_abandonment']),
                int(email['targeting_signals']['behavioral_triggers']['winback_campaign']),
                email['content_analysis']['email_length_chars']
            ))
        
        # Insert brand stats
        for brand, memory in self.data.get('brand_memory', {}).items():
            cursor.execute("""
                INSERT OR REPLACE INTO brand_stats VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                brand,
                memory['brand_metadata']['total_emails'],
                memory['brand_metadata']['first_email_date'],
                memory['brand_metadata']['last_email_date'],
                memory['pricing_patterns']['discount_statistics'].get('avg_discount'),
                memory['pricing_patterns']['discount_statistics'].get('max_discount'),
                memory['pricing_patterns']['discount_statistics'].get('min_discount'),
                memory['pricing_patterns']['discount_statistics']['discount_frequency'],
                memory['timing_patterns']['urgency_analysis']['avg_urgency_score'],
                memory['timing_patterns']['cyclical_patterns']['month_end_frequency'],
                memory['timing_patterns']['cyclical_patterns']['quarter_end_frequency'],
                memory['targeting_behavior']['personalization_frequency']
            ))
        
        conn.commit()
        conn.close()
        
        print("✓ Created SQLite database with emails and brand_stats tables")
    
    def _count_emails(self) -> int:
        """Count total emails in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
        conn.close()
        return count
    
    def _count_brands(self) -> int:
        """Count unique brands"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(DISTINCT brand_name) FROM emails").fetchone()[0]
        conn.close()
        return count
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    
    def query_to_sql(self, user_question: str) -> str:
        """Convert natural language question to SQL query using LLM"""
        
        schema_info = """
AVAILABLE TABLES:
1. emails - Individual email records
   Columns: email_id, brand_name, brand_domain, subject, received_date, 
            discount_value, has_discount, stacked_discounts, free_shipping, 
            coupon_code, urgency_score, exclusivity_score, personalization_score,
            has_urgency, urgency_type, flash_sale, countdown_timer,
            hour, day_of_week, day_of_month, month, quarter,
            is_weekend, is_month_end, is_quarter_end, is_holiday_period,
            personalized, cart_abandonment, winback_campaign, email_length

2. brand_stats - Aggregated brand statistics
   Columns: brand_name, total_emails, first_email_date, last_email_date,
            avg_discount, max_discount, min_discount, discount_frequency,
            avg_urgency_score, month_end_frequency, quarter_end_frequency,
            personalization_frequency
"""
        
        prompt = f"""Convert this user question to a SQLite query.

SCHEMA:
{schema_info}

USER QUESTION: {user_question}

RULES:
1. Return ONLY the SQL query, nothing else
2. Do NOT include "SQL", "SQLite", or any explanatory text
3. Do NOT wrap in markdown code blocks (no ```)
4. Use proper SQLite syntax
5. For counts, use COUNT(*)
6. For averages, use AVG()
7. For date filtering, use received_date
8. Brand names are case-sensitive, use LIKE for flexibility
9. Use LIMIT 10 for large result sets unless user asks for "all"

EXAMPLES:
Q: "How many emails from Target?"
A: SELECT COUNT(*) as email_count FROM emails WHERE brand_name LIKE '%Target%'

Q: "What's Nike's average discount?"
A: SELECT avg_discount FROM brand_stats WHERE brand_name LIKE '%Nike%'

Q: "Show me emails sent on weekends"
A: SELECT brand_name, subject, received_date FROM emails WHERE is_weekend = 1 LIMIT 10

Q: "Which brand emails me most?"
A: SELECT brand_name, COUNT(*) as count FROM emails GROUP BY brand_name ORDER BY count DESC LIMIT 5

Now convert this question to SQL (return ONLY the query):
{user_question}"""

        messages = [
            SystemMessage(content="You are a SQL expert. Return ONLY valid SQLite queries with no explanations, no markdown, no extra text."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        sql = response.content.strip()
        
        # Aggressive cleaning to extract SQL
        sql = self._extract_sql(sql)
        
        return sql

    def _extract_sql(self, text: str) -> str:
        """Extract SQL query from LLM response"""
        
        # Remove markdown code blocks
        if '```' in text:
            parts = text.split('```')
            for part in parts:
                part = part.strip()
                # Remove language identifiers
                for lang in ['sql', 'sqlite', 'SQL', 'SQLite']:
                    if part.startswith(lang):
                        part = part[len(lang):].strip()
                
                # Check if this looks like SQL
                if any(keyword in part.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']):
                    text = part
                    break
        
        # Split by newlines and find first SQL statement
        lines = text.split('\n')
        sql_lines = []
        found_select = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and common prefixes
            if not line:
                continue
            if line.lower().startswith(('here', 'sql:', 'query:', 'sqlite:', 'answer:', 'a:')):
                continue
                
            # Start capturing from SELECT
            if line.upper().startswith('SELECT') or found_select:
                found_select = True
                sql_lines.append(line)
                
                # Stop at semicolon or end of statement
                if line.endswith(';'):
                    break
        
        if sql_lines:
            sql = ' '.join(sql_lines)
        else:
            # Fallback: find first SELECT statement
            match = re.search(r'SELECT\s+.+', text, re.IGNORECASE | re.DOTALL)
            sql = match.group(0) if match else text
        
        # Final cleanup
        sql = sql.strip()
        
        # Remove trailing semicolon
        sql = sql.rstrip(';').strip()
        
        # Remove any remaining prefixes
        prefixes = ['sql', 'sqlite', 'query:', 'a:', 'answer:']
        for prefix in prefixes:
            if sql.lower().startswith(prefix):
                sql = sql[len(prefix):].strip()
        
        # Remove quotes around the entire query
        sql = sql.strip('"\'')
        
        # Fix smart quotes - replace curly quotes with straight quotes
        sql = sql.replace(''', "'").replace(''', "'")
        sql = sql.replace('"', '"').replace('"', '"')
        
        # Ensure proper quote characters for SQLite
        # Replace any remaining fancy quotes with standard single quotes
        sql = re.sub(r'[''‚`]', "'", sql)
        
        return sql
    
    def chat(self, question: str) -> str:
        """Answer questions using SQL + LLM interpretation"""
        
        print(f"\n💭 Question: {question}")
        
        try:
            # Step 1: Convert question to SQL
            print("🔍 Generating SQL query...")
            sql = self.query_to_sql(question)
            
            # Validate and fix SQL quotes
            sql = self._validate_sql_quotes(sql)
            
            print(f"📊 SQL: {sql}")
            
            # Step 2: Execute query
            print("⚡ Executing query...")
            results = self.execute_query(sql)
            
            # Step 3: Format results
            if results.empty:
                return "No results found for this query."
            
            results_str = results.to_string(index=False, max_rows=20)
            print(f"✓ Retrieved {len(results)} rows\n")
            
            # Step 4: Use LLM to interpret results
            interpretation_prompt = f"""Answer the user's question based on these query results.

USER QUESTION: {question}

SQL QUERY USED: {sql}

QUERY RESULTS:
{results_str}

Provide a clear, natural language answer. Include specific numbers and insights.
If the data shows trends or patterns, mention them."""

            messages = [
                SystemMessage(content="You are a helpful analyst. Provide clear, accurate answers based on data."),
                HumanMessage(content=interpretation_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error: {str(e)}\nTry rephrasing your question."
    
    def _validate_sql_quotes(self, sql: str) -> str:
        """Validate and fix SQL quote issues"""
        
        # Count single quotes - if odd number, there's likely a missing quote
        single_quote_count = sql.count("'")
        
        if single_quote_count % 2 != 0:
            # Odd number of quotes - simply add a closing quote at the end
            sql = sql + "'"
        
        # Fix double %% that might have been created
        sql = sql.replace("%%", "%")
        
        return sql
    
    def answer_mvp_questions(self) -> Dict[str, Any]:
        """Answer MVP questions using SQL queries"""
        
        print("\n" + "="*70)
        print("ANSWERING MVP QUESTIONS WITH SQL")
        print("="*70)
        
        answers = {}
        
        # Question 1: Discount aggression over time
        print("\n📊 Q1: Is this brand discounting more aggressively over time?")
        print("-"*70)
        
        q1_sql = """
        WITH ranked_emails AS (
            SELECT 
                brand_name,
                discount_value,
                received_date,
                ROW_NUMBER() OVER (PARTITION BY brand_name ORDER BY received_date) as email_num,
                COUNT(*) OVER (PARTITION BY brand_name) as total_count
            FROM emails 
            WHERE discount_value IS NOT NULL
        ),
        early_late AS (
            SELECT 
                brand_name,
                AVG(CASE WHEN email_num <= total_count/2 THEN discount_value END) as early_avg,
                AVG(CASE WHEN email_num > total_count/2 THEN discount_value END) as late_avg,
                COUNT(*) as sample_size
            FROM ranked_emails
            GROUP BY brand_name
            HAVING sample_size >= 2
        )
        SELECT 
            brand_name,
            ROUND(early_avg, 1) as early_discount,
            ROUND(late_avg, 1) as late_discount,
            ROUND(late_avg - early_avg, 1) as change,
            CASE 
                WHEN late_avg > early_avg + 5 THEN 'Increasing'
                WHEN late_avg < early_avg - 5 THEN 'Decreasing'
                ELSE 'Stable'
            END as trend
        FROM early_late
        ORDER BY change DESC
        """
        
        q1_results = self.execute_query(q1_sql)
        print(q1_results.to_string(index=False))
        answers['question_1'] = q1_results.to_dict('records')
        
        # Question 3: Cyclical clustering
        print("\n" + "="*70)
        print("📅 Q3: Do promotions cluster around quarter/month-end?")
        print("-"*70)
        
        q3_sql = """
        SELECT 
            brand_name,
            COUNT(*) as total_emails,
            SUM(is_month_end) as month_end_count,
            ROUND(100.0 * SUM(is_month_end) / COUNT(*), 1) as month_end_pct,
            SUM(is_quarter_end) as quarter_end_count,
            ROUND(100.0 * SUM(is_quarter_end) / COUNT(*), 1) as quarter_end_pct
        FROM emails
        GROUP BY brand_name
        HAVING total_emails >= 3
        ORDER BY month_end_pct DESC
        """
        
        q3_results = self.execute_query(q3_sql)
        print(q3_results.to_string(index=False))
        answers['question_3'] = q3_results.to_dict('records')
        
        # Question 4: Urgency authenticity
        print("\n" + "="*70)
        print("⚡ Q4: Is urgency language real or habitual?")
        print("-"*70)
        
        q4_sql = """
        SELECT 
            brand_name,
            COUNT(*) as total_emails,
            ROUND(AVG(urgency_score), 2) as avg_urgency,
            ROUND(100.0 * SUM(has_urgency) / COUNT(*), 1) as urgency_frequency_pct,
            CASE 
                WHEN AVG(urgency_score) > 0.7 THEN 'Habitual'
                WHEN AVG(urgency_score) < 0.3 THEN 'Authentic'
                ELSE 'Mixed'
            END as assessment
        FROM emails
        GROUP BY brand_name
        HAVING total_emails >= 2
        ORDER BY avg_urgency DESC
        """
        
        q4_results = self.execute_query(q4_sql)
        print(q4_results.to_string(index=False))
        answers['question_4'] = q4_results.to_dict('records')
        
        # Save results
        with open('mvp_answers_sql.json', 'w') as f:
            json.dump(answers, f, indent=2)
        
        print("\n" + "="*70)
        print("✓ Results saved to mvp_answers_sql.json")
        print("="*70)
        
        return answers
    
    def get_brand_summary(self, brand_name: str) -> Dict:
        """Get comprehensive summary for a specific brand"""
        
        sql = f"""
        SELECT * FROM brand_stats 
        WHERE brand_name LIKE '%{brand_name}%'
        """
        return self.execute_query(sql).to_dict('records')


def main():
    """Run the SQL-powered analyst"""
    
    print("="*70)
    print("SQL-POWERED INTELLIGENCE ANALYST")
    print("="*70)
    print()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("✗ Error: GOOGLE_API_KEY not set")
        return
    
    analyst = SQLIntelligenceAnalyst("email_intelligence.json")
    
    print("\nChoose mode:")
    print("1. Answer MVP questions (SQL-based)")
    print("2. Chat mode (SQL + LLM)")
    print("3. Both")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice in ["1", "3"]:
        analyst.answer_mvp_questions()
    
    if choice in ["2", "3"]:
        print("\n" + "="*70)
        print("CHAT MODE - Ask anything about your emails!")
        print("="*70)
        print("\nExamples:")
        print("  • How many emails did I get from Target?")
        print("  • What's the average discount across all brands?")
        print("  • Which brand sends the most weekend emails?")
        print("  • Show me all emails with discounts over 40%")
        print("  • Which brands use fake urgency language?")
        print("\nType 'quit' to exit\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            answer = analyst.chat(question)
            print(f"\n🤖 {answer}\n")


if __name__ == "__main__":
    main()