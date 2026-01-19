"""
Email Intelligence Agent using LangGraph and Gemini 2.5 Flash
Analyzes promotional emails to extract pricing, timing, language, and targeting signals
"""

import json
import os
import re
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Literal
from collections import defaultdict
import statistics

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

# =======================
# STATE DEFINITION
# =======================

class AgentState(TypedDict):
    """State for the email intelligence agent"""
    # Input
    raw_emails: List[Dict[str, Any]]
    
    # Processing
    current_index: int
    processed_emails: List[Dict[str, Any]]
    
    # Analysis
    brand_memory: Dict[str, Any]
    insights: Dict[str, Any]
    
    # Control
    status: Literal["processing", "analyzing", "complete", "error"]
    error: str

# =======================
# INITIALIZE LLM
# =======================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# =======================
# UTILITY FUNCTIONS
# =======================

def extract_brand_info(from_email: str) -> Dict[str, Any]:
    """Extract brand information from email sender"""
    # Extract brand name and domain
    match = re.search(r'<(.+@(.+))>', from_email)
    if match:
        full_email = match.group(1)
        domain = match.group(2)
        # Extract brand name before email
        brand_name = from_email.split('<')[0].strip().strip('"')
    else:
        full_email = from_email
        domain = from_email.split('@')[-1] if '@' in from_email else "unknown"
        brand_name = from_email.split('@')[0] if '@' in from_email else from_email
    
    return {
        "name": brand_name,
        "domain": domain,
        "category": "retail",  # Can be enhanced with LLM
        "parent_company": None
    }

def parse_datetime(date_str: str) -> Dict[str, Any]:
    """Parse email date and extract timing features"""
    try:
        # Parse the date string
        dt = datetime.strptime(date_str.split(' (')[0], "%a, %d %b %Y %H:%M:%S %z")
    except:
        try:
            dt = datetime.strptime(date_str.rsplit(' ', 1)[0], "%a, %d %b %Y %H:%M:%S")
        except:
            dt = datetime.now()
    
    # Calculate various timing features
    day_of_month = dt.day
    days_in_month = 31  # Simplified
    month = dt.month
    quarter = (month - 1) // 3 + 1
    
    return {
        "iso": dt.isoformat(),
        "hour": dt.hour,
        "day_of_week": dt.strftime("%A"),
        "day_of_month": day_of_month,
        "week_of_month": (day_of_month - 1) // 7 + 1,
        "month": month,
        "quarter": f"Q{quarter}",
        "is_weekend": dt.weekday() >= 5,
        "is_month_end": day_of_month > 25,
        "is_quarter_end": month % 3 == 0 and day_of_month > 25,
        "is_year_end": month == 12 and day_of_month > 20,
        "is_holiday_period": month in [11, 12] or (month == 1 and day_of_month < 7),
        "days_to_month_end": days_in_month - day_of_month,
        "days_to_quarter_end": (3 - (month - 1) % 3) * 30 - day_of_month
    }

def analyze_subject_line(subject: str) -> Dict[str, Any]:
    """Analyze subject line characteristics"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    
    emojis = emoji_pattern.findall(subject)
    
    return {
        "has_emoji": len(emojis) > 0,
        "emoji_count": len(emojis),
        "exclamation_marks": subject.count('!'),
        "question_marks": subject.count('?'),
        "all_caps_words": len(re.findall(r'\b[A-Z]{2,}\b', subject)),
        "number_present": bool(re.search(r'\d', subject)),
        "discount_in_subject": bool(re.search(r'\d+%\s*(off|discount)', subject, re.I))
    }

# =======================
# LANGGRAPH NODE FUNCTIONS
# =======================

def load_emails_node(state: AgentState) -> AgentState:
    """Load emails from JSON file"""
    
    # ⚙️ CONTROL LIMIT HERE - Set to None to process all emails
    EMAIL_LIMIT = 100  # Change this number or set to None for all emails
    
    try:
        with open('promotional_emails.json', 'r', encoding='utf-8') as f:
            emails = json.load(f)
        
        # Apply limit if specified
        if EMAIL_LIMIT is not None:
            emails = emails[:EMAIL_LIMIT]
            print(f"⚠️  Testing mode: Processing only first {EMAIL_LIMIT} emails")
        
        state['raw_emails'] = emails
        state['current_index'] = 0
        state['processed_emails'] = []
        state['brand_memory'] = {}
        state['status'] = 'processing'
        print(f"✓ Loaded {len(emails)} emails")
        
    except Exception as e:
        state['status'] = 'error'
        state['error'] = f"Failed to load emails: {str(e)}"
    
    return state

def extract_signals_node(state: AgentState) -> AgentState:
    """Extract signals from current email using Gemini"""
    
    if state['current_index'] >= len(state['raw_emails']):
        state['status'] = 'analyzing'
        return state
    
    email = state['raw_emails'][state['current_index']]
    
    try:
        # Prepare email content for LLM
        email_content = f"""
SUBJECT: {email.get('subject', 'N/A')}
FROM: {email.get('from', 'N/A')}
DATE: {email.get('date', 'N/A')}
SNIPPET: {email.get('snippet', 'N/A')}
BODY (first 1500 chars): {email.get('body', '')[:1500]}
"""
        
        # Create extraction prompt
        extraction_prompt = f"""You are an expert email marketing analyst. Analyze this promotional email and extract structured signals.

{email_content}

Extract the following in JSON format:

1. PRICING SIGNALS:
   - discount_values: List of all percentage or dollar discounts mentioned (e.g., [50, 20])
   - discount_text: List of exact discount phrases found
   - free_shipping: boolean
   - coupon_code: string or null
   - stacked_discounts: boolean (multiple discounts that stack)

2. TIMING SIGNALS:
   - urgency_phrases: List of urgency phrases (e.g., ["24 hours", "ending soon", "last chance"])
   - explicit_deadline: Any mentioned end date/time or null
   - countdown_timer: boolean
   - limited_quantity: boolean
   - flash_sale: boolean

3. LANGUAGE SIGNALS (0-1 scores):
   - urgency_score: How urgent does the language feel?
   - exclusivity_score: How exclusive/VIP does it feel?
   - personalization_score: How personalized is it?
   - action_verbs: List of action words (e.g., ["shop", "save", "grab"])
   - emotional_triggers: List of emotional words/phrases
   - power_words: List of impactful words

4. TARGETING SIGNALS:
   - personalized: boolean (uses name, references past behavior)
   - personalization_elements: List of personalization types
   - product_categories: List of product categories mentioned
   - cart_abandonment: boolean
   - winback_campaign: boolean
   - loyalty_mention: boolean

Return ONLY valid JSON with these exact keys. Be precise and thorough."""

        # Call Gemini
        messages = [
            SystemMessage(content="You are a precise email marketing analyst. Always return valid JSON."),
            HumanMessage(content=extraction_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Parse LLM response
        response_text = response.content.strip()
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        
        llm_signals = json.loads(response_text)
        
        # Build processed email record
        brand_info = extract_brand_info(email.get('from', ''))
        timing_info = parse_datetime(email.get('date', ''))
        subject_analysis = analyze_subject_line(email.get('subject', ''))
        
        # Construct pricing signals
        discount_values = llm_signals.get('discount_values', [])
        pricing_signals = {
            "has_discount": len(discount_values) > 0,
            "discount_type": "percentage" if discount_values else None,
            "primary_discount_value": max(discount_values) if discount_values else None,
            "additional_discounts": discount_values[1:] if len(discount_values) > 1 else [],
            "stacked_discounts": llm_signals.get('stacked_discounts', False),
            "effective_total_discount": sum(discount_values) if discount_values else None,
            "discount_text_extracted": llm_signals.get('discount_text', []),
            "original_prices_mentioned": [],
            "sale_prices_mentioned": [],
            "free_shipping": llm_signals.get('free_shipping', False),
            "minimum_purchase_amount": None,
            "coupon_code": llm_signals.get('coupon_code'),
            "price_anchoring": False,
            "msrp_comparison": False,
            "value_proposition": "deep_discount" if discount_values and max(discount_values) >= 40 else "moderate_discount"
        }
        
        # Construct timing signals
        urgency_phrases = llm_signals.get('urgency_phrases', [])
        timing_signals = {
            "has_urgency": len(urgency_phrases) > 0,
            "urgency_type": "time_limited" if urgency_phrases else "none",
            "urgency_phrases": urgency_phrases,
            "explicit_deadline": llm_signals.get('explicit_deadline'),
            "implied_deadline": None,
            "countdown_timer": llm_signals.get('countdown_timer', False),
            "limited_quantity_claim": llm_signals.get('limited_quantity', False),
            "flash_sale": llm_signals.get('flash_sale', False),
            "send_datetime": timing_info
        }
        
        # Construct language signals
        language_signals = {
            "urgency_score": llm_signals.get('urgency_score', 0),
            "exclusivity_score": llm_signals.get('exclusivity_score', 0),
            "personalization_score": llm_signals.get('personalization_score', 0),
            "scarcity_indicators": urgency_phrases,
            "urgency_words": urgency_phrases,
            "action_verbs": llm_signals.get('action_verbs', []),
            "emotional_triggers": llm_signals.get('emotional_triggers', []),
            "power_words": llm_signals.get('power_words', []),
            "fomo_language": urgency_phrases,
            "social_proof": [],
            "subject_line_analysis": subject_analysis,
            "tone": "promotional"
        }
        
        # Construct targeting signals
        targeting_signals = {
            "personalized": llm_signals.get('personalized', False),
            "personalization_elements": llm_signals.get('personalization_elements', []),
            "segment_indicators": [],
            "product_categories_mentioned": llm_signals.get('product_categories', []),
            "behavioral_triggers": {
                "cart_abandonment": llm_signals.get('cart_abandonment', False),
                "browse_abandonment": False,
                "winback_campaign": llm_signals.get('winback_campaign', False),
                "post_purchase": False,
                "loyalty_reward": llm_signals.get('loyalty_mention', False),
                "reactivation": False
            },
            "inferred_customer_status": "general_list"
        }
        
        # Create processed email record
        processed_email = {
            "email_id": email.get('id'),
            "extraction_timestamp": datetime.now().isoformat(),
            "raw_metadata": {
                "subject": email.get('subject'),
                "from": email.get('from'),
                "received_date": email.get('date'),
                "snippet": email.get('snippet')
            },
            "brand_info": brand_info,
            "pricing_signals": pricing_signals,
            "timing_signals": timing_signals,
            "language_signals": language_signals,
            "targeting_signals": targeting_signals,
            "content_analysis": {
                "primary_cta": None,
                "cta_count": 0,
                "links_count": len(re.findall(r'https?://', email.get('body', ''))),
                "has_images": False,
                "has_video": False,
                "product_count": 0,
                "social_proof_elements": [],
                "guarantee_mentioned": False,
                "return_policy_mentioned": False,
                "email_length_chars": len(email.get('body', '')),
                "body_truncated": len(email.get('body', '')) > 1500
            }
        }
        
        state['processed_emails'].append(processed_email)
        state['current_index'] += 1
        
        print(f"✓ Processed email {state['current_index']}/{len(state['raw_emails'])}: {brand_info['name']}")
        
    except Exception as e:
        print(f"✗ Error processing email {state['current_index']}: {str(e)}")
        state['current_index'] += 1  # Skip this email
    
    return state

def build_brand_memory_node(state: AgentState) -> AgentState:
    """Build long-term brand memory from processed emails"""
    
    print("\n Building brand memory...")
    
    brand_data = defaultdict(lambda: {
        "emails": [],
        "discounts": [],
        "urgency_scores": [],
        "personalization_scores": [],
        "send_hours": [],
        "send_days": [],
        "month_end_count": 0,
        "quarter_end_count": 0
    })
    
    # Aggregate data by brand
    for email in state['processed_emails']:
        brand_name = email['brand_info']['name']
        bd = brand_data[brand_name]
        
        bd['emails'].append(email)
        
        # Collect pricing data
        if email['pricing_signals']['primary_discount_value']:
            bd['discounts'].append(email['pricing_signals']['primary_discount_value'])
        
        # Collect language scores
        bd['urgency_scores'].append(email['language_signals']['urgency_score'])
        bd['personalization_scores'].append(email['language_signals']['personalization_score'])
        
        # Collect timing data
        bd['send_hours'].append(email['timing_signals']['send_datetime']['hour'])
        bd['send_days'].append(email['timing_signals']['send_datetime']['day_of_week'])
        
        if email['timing_signals']['send_datetime']['is_month_end']:
            bd['month_end_count'] += 1
        if email['timing_signals']['send_datetime']['is_quarter_end']:
            bd['quarter_end_count'] += 1
    
    # Build brand memory structure
    brand_memory = {}
    
    for brand_name, bd in brand_data.items():
        emails = bd['emails']
        total_emails = len(emails)
        
        # Sort by date
        emails.sort(key=lambda x: x['raw_metadata']['received_date'])
        
        first_date = emails[0]['raw_metadata']['received_date']
        last_date = emails[-1]['raw_metadata']['received_date']
        
        # Calculate discount statistics
        discounts = bd['discounts']
        discount_stats = {
            "avg_discount": statistics.mean(discounts) if discounts else None,
            "median_discount": statistics.median(discounts) if discounts else None,
            "max_discount": max(discounts) if discounts else None,
            "min_discount": min(discounts) if discounts else None,
            "discount_frequency": len(discounts) / total_emails if total_emails > 0 else 0
        }
        
        # Helper function to count occurrences
        def count_items(items):
            """Count occurrences and return as regular dict with int keys"""
            from collections import Counter
            counts = Counter(items)
            # Convert all keys and values to native Python types
            return {str(k): int(v) for k, v in counts.items()}
        
        # Build brand memory entry
        brand_memory[brand_name] = {
            "brand_metadata": {
                "first_email_date": first_date,
                "last_email_date": last_date,
                "total_emails": total_emails,
                "avg_email_frequency_days": None  # Would need date parsing
            },
            "pricing_patterns": {
                "discount_statistics": discount_stats,
                "discount_trend": {
                    "direction": "insufficient_data" if total_emails < 3 else "stable",
                    "slope": None
                }
            },
            "timing_patterns": {
                "send_time_distribution": {
                    "by_hour": count_items(bd['send_hours']) if bd['send_hours'] else {},
                    "by_day_of_week": count_items(bd['send_days']) if bd['send_days'] else {}
                },
                "urgency_analysis": {
                    "avg_urgency_score": statistics.mean(bd['urgency_scores']) if bd['urgency_scores'] else 0
                },
                "cyclical_patterns": {
                    "month_end_frequency": bd['month_end_count'] / total_emails if total_emails > 0 else 0,
                    "quarter_end_frequency": bd['quarter_end_count'] / total_emails if total_emails > 0 else 0
                }
            },
            "language_evolution": {
                "avg_urgency_score": statistics.mean(bd['urgency_scores']) if bd['urgency_scores'] else 0,
                "avg_personalization_score": statistics.mean(bd['personalization_scores']) if bd['personalization_scores'] else 0
            },
            "targeting_behavior": {
                "personalization_frequency": sum(1 for e in emails if e['targeting_signals']['personalized']) / total_emails if total_emails > 0 else 0
            }
        }
    
    state['brand_memory'] = brand_memory
    print(f"✓ Built memory for {len(brand_memory)} brands")
    
    return state

def generate_insights_node(state: AgentState) -> AgentState:
    """Generate strategic insights using Gemini"""
    
    print("\n🧠 Generating strategic insights...")
    
    # Prepare data summary for LLM
    summary = {
        "total_emails": len(state['processed_emails']),
        "unique_brands": len(state['brand_memory']),
        "brand_summaries": {}
    }
    
    for brand, memory in state['brand_memory'].items():
        summary['brand_summaries'][brand] = {
            "total_emails": memory['brand_metadata']['total_emails'],
            "avg_discount": memory['pricing_patterns']['discount_statistics']['avg_discount'],
            "avg_urgency_score": memory['language_evolution']['avg_urgency_score'],
            "month_end_frequency": memory['timing_patterns']['cyclical_patterns']['month_end_frequency'],
            "quarter_end_frequency": memory['timing_patterns']['cyclical_patterns']['quarter_end_frequency']
        }
    
    # Create insights prompt
    insights_prompt = f"""You are a strategic email marketing intelligence analyst. Analyze this brand behavior data and answer the MVP questions.

DATA SUMMARY:
{json.dumps(summary, indent=2)}

ANSWER THESE QUESTIONS in JSON format:

1. discount_aggression:
   - For each brand, is discounting increasing over time?
   - Which brands are most aggressive?
   - Overall trend?

2. post_inactivity_targeting:
   - Do any brands show patterns of changing behavior?
   - (Note: We don't have user engagement data yet, so note this limitation)

3. cyclical_clustering:
   - Do promotions cluster around month-end or quarter-end?
   - Which brands show this pattern most?
   - Clustering score 0-1?

4. urgency_authenticity:
   - Which brands overuse urgency language?
   - Is urgency real or habitual for each brand?
   - Authenticity score 0-1 for each brand?

Return JSON with this structure:
{{
  "discount_aggression": {{
    "overall_trend": "increasing/decreasing/stable",
    "evidence": ["list", "of", "evidence"],
    "brand_rankings": ["brand1", "brand2"]
  }},
  "post_inactivity_targeting": {{
    "data_limitation": "No user engagement data available yet",
    "recommendation": "Track open/click rates to analyze this"
  }},
  "cyclical_clustering": {{
    "pattern_detected": true/false,
    "clustering_score": 0.0-1.0,
    "evidence": ["list"],
    "top_brands": ["brand1"]
  }},
  "urgency_authenticity": {{
    "habitual_users": ["brands that overuse"],
    "authentic_users": ["brands that use sparingly"],
    "brand_scores": {{"Brand": 0.8}}
  }}
}}"""

    try:
        messages = [
            SystemMessage(content="You are a strategic analyst. Return only valid JSON."),
            HumanMessage(content=insights_prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Clean response
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        
        insights = json.loads(response_text)
        state['insights'] = insights
        
        print("✓ Generated strategic insights")
        
    except Exception as e:
        print(f"✗ Error generating insights: {str(e)}")
        state['insights'] = {
            "error": str(e),
            "note": "Failed to generate insights"
        }
    
    state['status'] = 'complete'
    return state

def save_results_node(state: AgentState) -> AgentState:
    """Save processed data and insights to JSON"""
    
    print("\n💾 Saving results...")
    
    output = {
        "schema_version": "1.0",
        "metadata": {
            "extraction_timestamp": datetime.now().isoformat(),
            "total_emails_processed": len(state['processed_emails']),
            "unique_brands": len(state['brand_memory'])
        },
        "processed_emails": state['processed_emails'],
        "brand_memory": state['brand_memory'],
        "strategic_insights": state['insights']
    }
    
    with open('email_intelligence.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("✓ Saved to email_intelligence.json")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Processed: {len(state['processed_emails'])} emails")
    print(f"Brands analyzed: {len(state['brand_memory'])}")
    print(f"\nTop brands by email volume:")
    sorted_brands = sorted(
        state['brand_memory'].items(),
        key=lambda x: x[1]['brand_metadata']['total_emails'],
        reverse=True
    )
    for brand, data in sorted_brands[:5]:
        count = data['brand_metadata']['total_emails']
        avg_disc = data['pricing_patterns']['discount_statistics']['avg_discount']
        print(f"  • {brand}: {count} emails, avg {avg_disc:.0f}% off" if avg_disc else f"  • {brand}: {count} emails")
    
    return state

# =======================
# BUILD LANGGRAPH
# =======================

def build_graph():
    """Build the LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_emails", load_emails_node)
    workflow.add_node("extract_signals", extract_signals_node)
    workflow.add_node("build_memory", build_brand_memory_node)
    workflow.add_node("generate_insights", generate_insights_node)
    workflow.add_node("save_results", save_results_node)
    
    # Define edges
    workflow.set_entry_point("load_emails")
    
    workflow.add_edge("load_emails", "extract_signals")
    
    # Loop through emails
    workflow.add_conditional_edges(
        "extract_signals",
        lambda state: "continue" if state['current_index'] < len(state['raw_emails']) else "done",
        {
            "continue": "extract_signals",
            "done": "build_memory"
        }
    )
    
    workflow.add_edge("build_memory", "generate_insights")
    workflow.add_edge("generate_insights", "save_results")
    workflow.add_edge("save_results", END)
    
    return workflow.compile()

# =======================
# MAIN EXECUTION
# =======================

def main():
    """Run the email intelligence agent"""
    
    print("="*60)
    print("EMAIL INTELLIGENCE AGENT")
    print("Powered by LangGraph + Gemini 2.5 Flash")
    print("="*60)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n✗ Error: GOOGLE_API_KEY environment variable not set")
        print("Set it with: export GOOGLE_API_KEY='your-key-here'")
        return
    
    # Build and run graph
    app = build_graph()
    
    initial_state = {
        "raw_emails": [],
        "current_index": 0,
        "processed_emails": [],
        "brand_memory": {},
        "insights": {},
        "status": "processing",
        "error": ""
    }
    
    try:
        result = app.invoke(initial_state)
        
        if result['status'] == 'complete':
            print("\n✓ Analysis complete! Check email_intelligence.json")
        elif result['status'] == 'error':
            print(f"\n✗ Error: {result['error']}")
            
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()