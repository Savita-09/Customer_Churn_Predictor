import os
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import traceback


class AIBusinessAnalyst:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize AI Business Analyst with customer churn data
        """
        self.df = df.copy()
        self.api_key = os.getenv("GROQ_API_KEY")
        print(f"🔑 API Key detected: {'✅ Yes' if self.api_key else '❌ No'}")
        
        self.client = None
        if self.api_key:
            try:
                print("🚀 Initializing Groq Chat client...")
                self.client = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name="openai/gpt-oss-120b",  
                    temperature=0.1,
                    max_tokens=1500
                )
                print("✅ Groq client ready!")
            except Exception as e:
                print(f"❌ Groq init failed: {e}")
                self.client = None
        else:
            print("⚠️ No API key - AI disabled")

    def get_data_summary(self) -> str:
        """Generate comprehensive, safe data summary"""
        try:
            summary_parts = []
            summary_parts.append(f"📊 **DATA OVERVIEW**")
            summary_parts.append(f"Total rows: {len(self.df)}")
            summary_parts.append(f"Columns: {len(self.df.columns)}")
            
            if 'Customer_Status' in self.df.columns:
                churn_rate = (self.df['Customer_Status'].astype(str).str.lower() == 'yes').mean() * 100
                summary_parts.append(f"Churn Rate: {churn_rate:.1f}%")
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:3]:  
                    mean_val = self.df[col].mean()
                    summary_parts.append(f"{col}: avg=${mean_val:.1f}")
            
            cat_cols = self.df.select_dtypes(include=['object']).columns
            for col in cat_cols[:3]: 
                top_val = self.df[col].value_counts().index[0]
                top_count = self.df[col].value_counts().iloc[0]
                summary_parts.append(f"{col}: top={top_val} ({top_count})")
            
            summary_parts.append(f"Columns: {list(self.df.columns)}")
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"SUMMARY ERROR: {str(e)}"

    def query(self, user_question: str) -> str:
        """
        Main query method - FIXED to always return proper response
        """
        print(f"🤖 Processing query: {user_question[:50]}...")
        
        if not self.client:
            return """
**🚫 AI UNAVAILABLE**
No Groq API key detected. 

**Quick Fix:**
1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Create FREE API key  
3. Paste in Streamlit sidebar
4. Refresh page 🔄
            """

        try:
            data_context = self.get_data_summary()
  
            system_prompt = """You are ChurnGuard AI, a telecom churn expert. Analyze customer data and give actionable insights.

RULES:
- Reference the DATA CONTEXT
- Be specific and quantitative
- Use bullet points •
- End with 3 PRIORITY ACTIONS
- Keep responses < 800 words

FORMAT:
**Insight** | **Why it matters** | **Action**
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
**DATA CONTEXT:**
{data_context}

**QUESTION:** {user_question}

Provide expert analysis:
                """)
            ]
            
            print("📡 Sending to Groq...")
            
            response = self.client.invoke(messages)
            
            if hasattr(response, 'content') and response.content:
                result = response.content.strip()
                print(f"✅ Response received: {len(result)} chars")
                return result
            else:
                print("❌ Empty response from Groq")
                return "**🤖 No response** - Try rephrasing your question"
                
        except Exception as e:
            error_details = str(e)
            print(f"❌ Query failed: {error_details}")
            print(traceback.format_exc())
            
            fallback_responses = {
                "rate limit": """
**⏳ RATE LIMITED**
Groq quota exceeded. Wait 1-2 mins or upgrade plan.
                """,
                "invalid": """
**🔑 API ISSUE**
Invalid API key. Get new one from console.groq.com
                """,
                "network": """
**🌐 NETWORK ERROR**
Check internet. Try again in 30 seconds.
                """
            }
            
            for key, msg in fallback_responses.items():
                if key in error_details.lower():
                    return msg
            
            return f"""
**⚠️ ANALYSIS ERROR**
Technical issue: {error_details[:100]}...

**Try:**
- Refresh page
- Simpler question
- Check API quota
            """

    def quick_analysis(self, analysis_type: str = "churn_drivers") -> str:
        """Quick predefined analyses"""
        prompts = {
            "churn_drivers": "What are the top 3 churn drivers and retention priorities?",
            "revenue_risk": "Which customer segments pose highest revenue risk?",
            "retention_plan": "Create 30-day retention action plan"
        }
        return self.query(prompts.get(analysis_type, prompts["churn_drivers"]))


if __name__ == "__main__":
    test_df = pd.DataFrame({
        'Tenure': np.random.randint(1, 73, 1000),
        'Monthly_Charges': np.random.uniform(20, 120, 1000),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
        'Customer_Status': np.random.choice(['Yes', 'No'], 1000, p=[0.26, 0.74])
    })
    
    analyst = AIBusinessAnalyst(test_df)
    print("=== DATA SUMMARY ===")
    print(analyst.get_data_summary())
    print("\n=== TEST QUERY ===")
    print(analyst.query("What drives churn?"))