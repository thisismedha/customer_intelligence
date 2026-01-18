"""
Intelligence Analyst Agent - Reads processed data and answers questions
Separates data processing from intelligence analysis for faster iteration
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class IntelligenceAnalyst:
    """Agent that reads email intelligence and answers questions"""
    
    def __init__(self, intelligence_file: str = "email_intelligence.json"):
        """Initialize analyst with processed intelligence data"""
        
        print("🤖 Initializing Intelligence Analyst Agent...")
        
        # Load intelligence data
        with open(intelligence_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,  # Slightly higher for conversational responses
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Conversation history for context
        self.conversation_history = []
        
        # Extract summary stats
        self.stats = {
            "total_emails": len(self.data.get('processed_emails', [])),
            "total_brands": len(self.data.get('brand_memory', {})),
            "brands": list(self.data.get('brand_memory', {}).keys())
        }
        
        print(f"✓ Loaded intelligence on {self.stats['total_emails']} emails")
        print(f"✓ Analyzing {self.stats['total_brands']} brands: {', '.join(self.stats['brands'][:3])}...")
        print()
    
    def answer_mvp_questions(self) -> Dict[str, Any]:
        """Answer the 4 core MVP questions"""
        
        print("📊 Analyzing data to answer MVP questions...\n")
        
        # Prepare data summary
        brand_summaries = {}
        for brand, memory in self.data['brand_memory'].items():
            brand_summaries[brand] = {
                "total_emails": memory['brand_metadata']['total_emails'],
                "discount_stats": memory['pricing_patterns']['discount_statistics'],
                "urgency_stats": memory['timing_patterns']['urgency_analysis'],
                "cyclical_patterns": memory['timing_patterns']['cyclical_patterns']
            }
        
        # Get detailed email data for trend analysis
        emails_by_brand = {}
        for email in self.data['processed_emails']:
            brand = email['brand_info']['name']
            if brand not in emails_by_brand:
                emails_by_brand[brand] = []
            emails_by_brand[brand].append({
                "date": email['raw_metadata']['received_date'],
                "discount": email['pricing_signals']['primary_discount_value'],
                "urgency_score": email['language_signals']['urgency_score'],
                "is_month_end": email['timing_signals']['send_datetime']['is_month_end'],
                "is_quarter_end": email['timing_signals']['send_datetime']['is_quarter_end']
            })
        
        # Sort emails by date for trend analysis
        for brand in emails_by_brand:
            emails_by_brand[brand].sort(key=lambda x: x['date'])
        
        analysis_prompt = f"""You are an expert email marketing intelligence analyst. Answer these 4 questions based on the data.

BRAND SUMMARIES:
{json.dumps(brand_summaries, indent=2)}

DETAILED EMAIL PROGRESSION BY BRAND:
{json.dumps(emails_by_brand, indent=2)}

Answer these 4 questions with specific evidence:

QUESTION 1: Is this brand discounting more aggressively over time?
- For EACH brand, compare early discounts to recent discounts
- Identify clear trends (e.g., "Started 30%, now 50%")
- Score aggression 0-1 (0=never discounts, 1=constantly deep discounts)

QUESTION 2: Does this brand target me differently after inactivity?
- Acknowledge: No user engagement data available yet
- But note: If email frequency/discounts change mid-sequence, mention it
- Recommend: Add open/click tracking to answer properly

QUESTION 3: Do promotions cluster around quarter-end / inventory cycles?
- Calculate % of emails in last 7 days of month
- Calculate % in last 2 weeks of quarter
- >50% = significant clustering
- Identify which brands do this

QUESTION 4: Is urgency language real or habitual?
- If avg urgency_score > 0.7 consistently = habitual
- If urgency varies and is low on average = authentic
- Score 0-1 (0=always fake, 1=always real)

Return JSON:
{{
  "question_1": {{
    "by_brand": {{
      "Brand": {{
        "answer": "Yes - becoming more aggressive / No - stable / Insufficient data",
        "trend": "increasing/stable/decreasing",
        "evidence": "Started at X%, now at Y%",
        "aggression_score": 0.0-1.0,
        "first_discount": X,
        "latest_discount": Y
      }}
    }},
    "summary": "Overall finding across all brands"
  }},
  "question_2": {{
    "answer": "Cannot determine - no engagement data",
    "observable_patterns": "Any sequence changes noted",
    "recommendation": "Track opens/clicks to analyze post-inactivity behavior"
  }},
  "question_3": {{
    "by_brand": {{
      "Brand": {{
        "answer": "Yes/No",
        "month_end_pct": X,
        "quarter_end_pct": Y,
        "clustering_score": 0.0-1.0
      }}
    }},
    "summary": "Which brands cluster and why"
  }},
  "question_4": {{
    "by_brand": {{
      "Brand": {{
        "answer": "Real/Habitual/Mixed",
        "authenticity_score": 0.0-1.0,
        "avg_urgency": X,
        "evidence": "Specific pattern description"
      }}
    }},
    "summary": "Overall urgency authenticity findings"
  }}
}}"""

        try:
            messages = [
                SystemMessage(content="You are a precise analyst. Return valid JSON with evidence-based answers."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Clean markdown
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            answers = json.loads(response_text)
            
            # Save answers
            output = {
                "generated_at": datetime.now().isoformat(),
                "analysis_summary": self.stats,
                "mvp_answers": answers
            }
            
            with open('mvp_answers.json', 'w') as f:
                json.dump(output, f, indent=2)
            
            self._print_answers(answers)
            
            return answers
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return {"error": str(e)}
    
    def _print_answers(self, answers: Dict):
        """Pretty print answers to console"""
        
        print("\n" + "="*70)
        print("BRAND INTELLIGENCE REPORT")
        print("="*70)
        
        # Q1
        print("\n📊 Q1: Is this brand discounting more aggressively over time?")
        print("-"*70)
        if 'question_1' in answers:
            q1 = answers['question_1']
            print(f"Summary: {q1.get('summary', 'N/A')}\n")
            for brand, data in q1.get('by_brand', {}).items():
                print(f"• {brand}: {data.get('answer', 'N/A')}")
                print(f"  Evidence: {data.get('evidence', 'N/A')}")
                print(f"  Aggression Score: {data.get('aggression_score', 0):.2f}\n")
        
        # Q2
        print("="*70)
        print("📧 Q2: Does this brand target me differently after inactivity?")
        print("-"*70)
        if 'question_2' in answers:
            q2 = answers['question_2']
            print(f"Answer: {q2.get('answer', 'N/A')}")
            print(f"Recommendation: {q2.get('recommendation', 'N/A')}\n")
        
        # Q3
        print("="*70)
        print("📅 Q3: Do promotions cluster around quarter/month-end?")
        print("-"*70)
        if 'question_3' in answers:
            q3 = answers['question_3']
            print(f"Summary: {q3.get('summary', 'N/A')}\n")
            for brand, data in q3.get('by_brand', {}).items():
                print(f"• {brand}: {data.get('answer', 'N/A')}")
                print(f"  Month-end: {data.get('month_end_pct', 0):.1f}%")
                print(f"  Quarter-end: {data.get('quarter_end_pct', 0):.1f}%\n")
        
        # Q4
        print("="*70)
        print("⚡ Q4: Is urgency language real or habitual?")
        print("-"*70)
        if 'question_4' in answers:
            q4 = answers['question_4']
            print(f"Summary: {q4.get('summary', 'N/A')}\n")
            for brand, data in q4.get('by_brand', {}).items():
                print(f"• {brand}: {data.get('answer', 'N/A')}")
                print(f"  Authenticity Score: {data.get('authenticity_score', 0):.2f}")
                print(f"  Evidence: {data.get('evidence', 'N/A')}\n")
        
        print("="*70)
    
    def chat(self, question: str) -> str:
        """Conversational interface - ask any question about the data"""
        
        # Add user question to history
        self.conversation_history.append(HumanMessage(content=question))
        
        # Prepare context with data summary
        context = f"""You are an intelligent assistant analyzing email marketing data.

DATA AVAILABLE:
- {self.stats['total_emails']} emails analyzed
- {self.stats['total_brands']} brands tracked
- Brands: {', '.join(self.stats['brands'])}

FULL DATA:
{json.dumps(self.data, indent=2)[:5000]}  # First 5000 chars

Answer the user's question using this data. Be conversational, specific, and cite evidence."""

        messages = [
            SystemMessage(content=context),
            *self.conversation_history
        ]
        
        response = self.llm.invoke(messages)
        
        # Add AI response to history
        self.conversation_history.append(AIMessage(content=response.content))
        
        return response.content
    
    def compare_brands(self, brands: List[str], metric: str = "discount") -> Dict:
        """Compare specific brands on a metric"""
        
        comparison = {}
        
        for brand in brands:
            if brand in self.data['brand_memory']:
                memory = self.data['brand_memory'][brand]
                
                if metric == "discount":
                    comparison[brand] = memory['pricing_patterns']['discount_statistics']
                elif metric == "urgency":
                    comparison[brand] = memory['timing_patterns']['urgency_analysis']
                elif metric == "frequency":
                    comparison[brand] = memory['brand_metadata']['total_emails']
        
        return comparison
    
    def get_brand_timeline(self, brand: str) -> List[Dict]:
        """Get chronological timeline of emails for a brand"""
        
        timeline = []
        for email in self.data['processed_emails']:
            if email['brand_info']['name'] == brand:
                timeline.append({
                    "date": email['raw_metadata']['received_date'],
                    "subject": email['raw_metadata']['subject'],
                    "discount": email['pricing_signals']['primary_discount_value'],
                    "urgency_score": email['language_signals']['urgency_score']
                })
        
        timeline.sort(key=lambda x: x['date'])
        return timeline


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Run the Intelligence Analyst Agent"""
    
    print("="*70)
    print("INTELLIGENCE ANALYST AGENT v2.0")
    print("="*70)
    print()
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("✗ Error: GOOGLE_API_KEY not set")
        return
    
    # Initialize analyst
    analyst = IntelligenceAnalyst("email_intelligence.json")
    
    # Mode selection
    print("Choose mode:")
    print("1. Answer MVP questions (static)")
    print("2. Conversational mode (chat)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ["1", "3"]:
        print("\n" + "="*70)
        print("MODE 1: ANSWERING MVP QUESTIONS")
        print("="*70)
        answers = analyst.answer_mvp_questions()
    
    if choice in ["2", "3"]:
        print("\n" + "="*70)
        print("MODE 2: CONVERSATIONAL INTELLIGENCE")
        print("="*70)
        print("\nYou can now ask questions about your email data!")
        print("Examples:")
        print("  - Which brand gives me the best deals?")
        print("  - Show me Nike's discount trend")
        print("  - When does Victoria's Secret usually email me?")
        print("  - Compare Nike and Adidas on urgency")
        print("\nType 'quit' to exit\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = analyst.chat(user_input)
            print(f"\n🤖 Analyst: {response}\n")


if __name__ == "__main__":
    main()