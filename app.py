import streamlit as st
from streamlit.components.v1 import html

# Standard library imports
import os
import json
import traceback
import warnings
from datetime import datetime
import time
import random
from uuid import UUID, uuid4
from typing import Optional, Dict, Any, Tuple, List
from functools import lru_cache

# Third-party imports
import openai
from dotenv import load_dotenv
from supabase import create_client, Client
from streamlit_feedback import streamlit_feedback

# LangChain imports
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import HumanMessage, AIMessage, AgentAction, AgentFinish
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.manager import get_openai_callback


# citation imports
from conversation_starter_citations import CONVERSATION_STARTER_CITATIONS
from communication_strategies_citations import COMMUNICATION_STRATEGIES_CITATIONS
from simulation_citations import SIMULATION_CITATIONS
from Website_citations import WEBSITE_CITATIONS
from Active_listening_citations import ACTIVE_LISTENING_CITATIONS
from i_messages_citations import I_MESSAGES_CITATIONS
from positive_reinforcement import POSITIVE_REINFORCEMENT_CITATIONS
from Reflective_questioning import REFLECTIVE_QUESTIONING_CITATIONS

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set Streamlit page config
st.set_page_config(
    layout="wide",
    page_title="Parenting Support Bot",
    initial_sidebar_state="expanded"
)

def check_environment():
    """Check and initialize required environment variables"""
    load_dotenv()
    
    required_vars = [
        'OPENAI_API_KEY',
        'SUPABASE_URL',
        'SUPABASE_KEY',
        'LANGCHAIN_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not (os.getenv(var) or hasattr(st.secrets, var))]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        st.error(error_msg)
        raise EnvironmentError(error_msg)
    
    openai.api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY') or st.secrets.get('LANGCHAIN_API_KEY')

# Initialize LangSmith Client
# smith_client = Client()

def create_langsmith_run(name: str, inputs: Dict[str, Any], fallback_id: Optional[str] = None) -> str:
    """Create a unique identifier for tracking runs"""
    return str(uuid4())

def update_langsmith_run(run_id: str, outputs: Dict[str, Any]):
    """Update run with outputs"""
    pass

def setup_memory():
    """Sets up memory for the application"""
    return ConversationBufferWindowMemory(k=3)

# Initialize Memory
memory = setup_memory()

def inject_qualtrics_messenger():
    """Inject JavaScript code for Qualtrics communication"""
    return html("""
        <script>
        // Function to send messages to Qualtrics
        function notifyQualtrics(feature) {
            window.parent.postMessage({
                'feature': feature,
                'complete': true
            }, '*');
        }

        // Listen for feature changes
        window.addEventListener('featureChange', function(e) {
            notifyQualtrics(e.detail.feature);
        });
        </script>
    """)

CUSTOM_CSS = """
    <style>
    /* General Typography */
    body {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
    }

    h1, h2, h3, .section-header {
        font-family: 'Roboto', sans-serif;
        color: #2563eb;
    }

    .main-header {
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 1.5em;
        color: #2563eb;
    }

    .section-header {
        font-size: 2em;
        font-weight: 700;
        margin: 1.2em 0;
        border-bottom: 3px solid #60a5fa;
        padding-bottom: 0.5em;
    }

    .description-text {
        font-size: 1.1em;
        line-height: 1.8;
        margin: 1.2em 0;
    }

    /* Button Styling */
    div.stButton > button {
        height: 3em;
        font-size: 1em;
        font-weight: 500;
        background-color: #2563eb;
        color: white;
        border-radius: 0.5em;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Message Bubbles */
    .message-parent, .message-child {
        padding: 1.2em;
        border-radius: 1em;
        margin: 0.8em 0;
        max-width: 80%;
    }

    .message-parent {
        background-color: #60a5fa;
        color: white;
        margin-left: auto;
    }

    .message-child {
        background-color: #f3f4f6;
        color: #1a1a1a;
        margin-right: auto;
    }

    /* New Styles for UI Updates */
    .stSelectbox {
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
    }
    
    /* Situation Box Styling */
    .situation-box {
        background-color: #f0f7ff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Control Feedback Styling */
    .control-feedback {
        margin-top: 0.5rem;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .pause-feedback {
        background-color: #f0f7ff;
    }
    
    .hints-feedback {
        background-color: #f0f7ff;
    }
    
    .reformulate-feedback {
        background-color: #fff7e6;
    }

    /* Strategy Selection Styling */
    .strategy-select {
        margin: 1rem 0;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }

    /* Conversation Section Styling */
    .conversation-section {
        margin: 2rem 0;
    }

    .conversation-input {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    </style>
"""

# Strategy Explanations
STRATEGY_EXPLANATIONS = {
    "Active Listening": """
        <div class='strategy-explanation' style='background-color: #f3f4f6; border-left: 4px solid #2563eb;'>
            👂 <strong style='color: #2563eb; font-size: 1.2em;'>Active Listening</strong><br>
            <p style='font-size: 1.1em; line-height: 1.8;'>
                Fully focus on, understand, and remember what your child is saying. 
                This helps them feel heard and valued.
            </p>
        </div>
    """,
    "Positive Reinforcement": """
        <div class='strategy-explanation' style='background-color: #f3f4f6; border-left: 4px solid #2563eb;'>
            ⭐ <strong style='color: #2563eb; font-size: 1.2em;'>Positive Reinforcement</strong><br>
            <p style='font-size: 1.1em; line-height: 1.8;'>
                Encourage desired behaviors through specific praise or rewards, 
                helping build self-esteem and motivation.
            </p>
        </div>
    """,
    "Reflective Questioning": """
        <div class='strategy-explanation' style='background-color: #f3f4f6; border-left: 4px solid #2563eb;'>
            ❓ <strong style='color: #2563eb; font-size: 1.2em;'>Reflective Questioning</strong><br>
            <p style='font-size: 1.1em; line-height: 1.8;'>
                Use open-ended questions to help children think deeper and express themselves. 
                For example: 'What do you think about...?'
            </p>
        </div>
    """
}


# custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
class SupabaseManager:
    def __init__(self):
        if hasattr(st, 'secrets'):
            self.supabase_url = st.secrets.get('SUPABASE_URL')
            self.supabase_key = st.secrets.get('SUPABASE_KEY')
        else:
            self.supabase_url = os.getenv('SUPABASE_URL')
            self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase: Optional[Client] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize Supabase client with improved error handling"""
        print("\n=== Initializing Supabase Connection ===")
        if self._initialized:
            print("Already initialized")
            return True

        try:
            if not self.supabase_url or not self.supabase_key:
                raise ValueError("Missing Supabase credentials")
            
            print("Creating Supabase client...")
            self.supabase = create_client(
                supabase_url=self.supabase_url,
                supabase_key=self.supabase_key
            )
            
            print("Testing connection...")
            try:
                test_result = self.supabase.table('simulations').select("count", count='exact').limit(1).execute()
                print("Successfully connected to simulations table")
            except Exception as e:
                if "relation" in str(e) and "does not exist" in str(e):
                    print("Simulations table doesn't exist, will be created via SQL migration")
                else:
                    print(f"Error accessing simulations table: {str(e)}")
                    return False
            
            self._initialized = True
            print("Supabase initialization successful")
            return True
            
        except Exception as e:
            print(f"Supabase initialization error: {str(e)}")
            traceback.print_exc()
            return False

    def ensure_initialized(self) -> bool:
        """Helper method to ensure connection is initialized"""
        if not self._initialized and not self.initialize():
            print("Failed to initialize Supabase connection")
            return False
        return True

    # Simulation Methods
    def save_simulation_data(self, simulation_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Save simulation data with validation and error handling"""
        if not self.ensure_initialized():
            return False, "Failed to initialize Supabase connection"

        try:
            # Validate required fields
            required_fields = ['user_id', 'simulation_data', 'created_at']
            missing_fields = [f for f in required_fields if f not in simulation_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            # Ensure data is JSON serializable
            json.dumps(simulation_data['simulation_data'])

            data = {
                "user_id": simulation_data["user_id"],
                "simulation_data": simulation_data["simulation_data"],
                "created_at": simulation_data["created_at"],
                "completed_at": None,
                "langsmith_run_id": simulation_data.get("langsmith_run_id")
            }

            result = self.supabase.table('simulations').insert(data).execute()
            
            if not result.data:
                raise ValueError("No data returned from insert operation")

            return True, result.data[0].get('id')

        except Exception as e:
            error_msg = f"Error saving simulation data: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg

    def complete_simulation(self, simulation_id: str, langsmith_run_id: Optional[str] = None) -> Tuple[bool, str]:
        """Complete a simulation with improved error handling"""
        if not self.ensure_initialized():
            return False, "Failed to initialize Supabase connection"

        try:
            if not simulation_id:
                return False, "simulation_id is required"

            data = {
                "completed_at": datetime.utcnow().isoformat(),
                "langsmith_run_id": langsmith_run_id
            }

            result = self.supabase.table('simulations')\
                .update(data)\
                .eq('id', simulation_id)\
                .execute()

            if not result.data:
                return False, "No data returned from update operation"

            return True, "Simulation completed successfully"

        except Exception as e:
            error_msg = f"Error completing simulation: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg

    def view_simulations(self, user_id: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """View simulations with proper error handling"""
        if not self.ensure_initialized():
            return False, []

        try:
            query = self.supabase.table('simulations')
            if user_id:
                query = query.eq('user_id', user_id)
            result = query.execute()
            return True, result.data if result and result.data else []
        except Exception as e:
            print(f"Error fetching simulations: {e}")
            traceback.print_exc()
            return False, []
        
# Initialize Supabase manager
supabase_manager = SupabaseManager()

def setup_memory():
    """Sets up memory for the application"""
    return ConversationBufferWindowMemory(k=3)

# Initialize Memory
memory = setup_memory()

def init_session_state():
    session_vars = {
        'run_id': str(uuid4()),
        'agentState': "start",
        'consent': False,
        'exp_data': True,
        'llm_model': "gpt-4",
        'simulation_ended': False,
        'stored_responses': {},
        'info_submitted': False,
        'conversation_history': [],
        'child_mood': random.choice(['cooperative', 'defiant', 'distracted']),
        'turn_count': 0,
        'strategy': "Active Listening",
        'simulation_id': str(uuid4()),
        'visited_features': set(),
        'situation': "",  # Add this default value
        'feature_outputs': {
            'advice': {},
            'conversation_starters': {},
            'communication_techniques': {},
            'simulation_history': []
        }
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

@st.cache_data(ttl=3600)
def cached_openai_call(messages, model="gpt-4", temperature=0.7, max_tokens=150):
    try:
        completion = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None
            
def track_feature_visit(feature_name: str):
    """Track which features the user has visited"""
    if 'visited_features' not in st.session_state:
        st.session_state.visited_features = set()
    
    feature_display_map = {
        "advice": "Advice",
        "conversation_starters": "Conversation Starters",
        "communication_techniques": "Communication Techniques",
        "role_play": "Role-Play Simulation",
        "Role-Play Simulation": "Role-Play Simulation"
    }
    
    normalized_name = feature_display_map.get(feature_name, feature_name)
    st.session_state.visited_features.add(normalized_name)
    st.session_state.visited_features.add(normalized_name.lower().replace(" ", "_"))
    
    st.components.v1.html(f"""
        <script>
        window.dispatchEvent(new CustomEvent('featureChange', {{
            detail: {{ feature: '{normalized_name.lower().replace(" ", "_")}' }}
        }}));
        </script>
    """)

def generate_child_response(conversation_history, child_age, situation, mood, strategy, parent_response):
    """Generate a child's response based on the current context with improved personalization"""
    child_name = st.session_state.get('child_name', 'the child')
    response_key = f"{parent_response}_{child_age}_{mood}_{strategy}"
    
    if response_key in st.session_state['stored_responses']:
        return st.session_state['stored_responses'][response_key]
    
    # Format recent chat history
    recent_chat = ' | '.join([
        f"{msg['role']}: {msg['content']}" 
        for msg in conversation_history[-2:]
    ])
    
    messages = [
        {"role": "system", "content": f"""You are simulating {child_name}'s responses in a parent-child conversation.
        Core Parameters:
        - Child Name: {child_name}
        - Age Range: {child_age}
        - Current Mood: {mood}
        - Situation Context: {situation}
        - Recent Chat History: {recent_chat}
        
        Emotional State Guidelines:
        1. Cooperative Mood:
          - Demonstrates willingness to understand
          - Expresses interest in learning why, not just what
          - Shows engagement through relevant questions
          - Maintains age-appropriate responses

        2. Defiant Mood:
           - Display resistance through words and actions
           - Include age-appropriate emotional outbursts
           - Make comparisons to siblings/friends ("But Sarah gets to!")
           - Express feelings of unfairness
        
        3. Distracted Mood:
           - Show divided attention ("But wait, can I just...")
           - Reference current activities or interests
           - Demonstrate difficulty focusing on parent's words
        
        Response Format Rules:
        1. Keep responses concise (1-2 sentences maximum)
        2. Respond to the situation context specifically
        3. Use vocabulary appropriate for {child_age}
        4. Maintain character consistency
        5. React to the parent's specific words or approach
        6. Never explain rationally or break character
        
        Remember:
        - Stay firmly in character as {child_name}
        - Reflect the {mood} mood throughout the response
        - Incorporate details from the specific situation
        - Keep responses authentic to the age range
        - Never provide meta-commentary or explanations
        - Respond directly to parent's last message: "{parent_response}" """},
        {"role": "user", "content": f"Generate {child_name}'s next response:"}
    ]

    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=60
        )
        response = completion.choices[0].message.content.strip()
        
        st.session_state['stored_responses'][response_key] = response
        return response
    except Exception as e:
        print(f"Error generating child response: {e}")
        return "I don't know what to say..."

def provide_realtime_feedback(parent_response: str, strategy: str, situation: str, child_age: str, conversation_history: list) -> dict:
    """
    Provides two types of feedback:
    1. Concise, meaningful in-chat hints
    2. Detailed feedback for conversation playback
    """
    child_name = st.session_state.get('child_name', 'your child')
    child_mood = st.session_state.get('child_mood', 'neutral')
    
    messages = [
        {"role": "system", "content": f"""
        You are a parenting coach providing feedback in two parts:

        - One specific, meaningful observation about the parent's response
        - One clear, actionable suggestion tailored to the situation
        - Keep it under 20 words total
        - Make it relevant to their chosen strategy ({strategy})
        - No bullet points or labels
        Example: "Acknowledging teddy's importance. Try adding when you'll read the story tomorrow."

        - Detailed analysis (2-3 sentences)
        - Connect observation to the specific situation
        - Provide thorough improvement suggestion
        - Reference the chosen strategy and its effectiveness
        - Consider child's age and current mood
        
        Current Context:
        - Child: {child_name} (Age: {child_age})
        - Mood: {child_mood}
        - Strategy: {strategy}
        - Situation: {situation}
        """},
        {"role": "user", "content": f"Parent's response: {parent_response}\nProvide both types of feedback:"}
    ]

    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        feedback_parts = completion.choices[0].message.content.split("\n\n")
        
        feedback = {
            "hint": feedback_parts[0].strip(),
            "detailed": feedback_parts[1].strip()
        }
        
        return feedback
        
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return {
            "hint": f"Consider acknowledging {child_name}'s feelings while maintaining bedtime routine",
            "detailed": f"When using {strategy}, try to balance understanding with consistent boundaries."
        }
    
def generate_conversation_starters(situation: str) -> str:
    """Generate conversation starters based on the situation"""
    prompt = f"""
    SYSTEM
    Use the provided citations delimited by triple quotes to answer questions.
    USER
    Academic Citations:
    {CONVERSATION_STARTER_CITATIONS}

    Website Resources:
    {WEBSITE_CITATIONS}

    Question: Provide conversation starters for the following situation with a child: {situation}
    """
    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating conversation starters: {e}")
        return "Unable to generate conversation starters at this time."

def display_advice(parent_name: str, child_age: str, situation: str):
    """Display parenting advice in a visually engaging card layout with accessible colors"""
    st.markdown("<h2 class='section-header'>Parenting Advice</h2>", unsafe_allow_html=True)
    
    if not situation:
        st.warning("Please describe the situation to get advice.")
        return

    try:
        with st.spinner('Generating advice...'):
            messages = [
                {"role": "system", "content": f"""
                    You are a parenting expert. Generate 4 specific, practical pieces of advice for the given situation.
                    Each piece of advice should be:
                    - Detailed enough to be actionable (25-30 words)
                    - Specific to the situation and child's age
                    - Based on evidence-backed parenting strategies
                    - Include both what to do and why it works
                    
                    Format each piece of advice as a JSON object with these fields:
                    - title: A short 2-3 word summary
                    - advice: The detailed advice
                    - icon: A relevant emoji
                    - color: The background color code (use exactly these colors in order):
                      ["#2F6DA3", "#2A7A5E", "#A35E2F", "#6C596E"]
                    
                    Return exactly 4 items.
                """},
                {"role": "user", "content": f"Child age: {child_age}\nSituation: {situation}\nGenerate specific advice for handling this situation."}
            ]

            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            advice_list = json.loads(completion.choices[0].message.content)
            
            # Add custom CSS for softer shadows and better contrast
            st.markdown("""
                <style>
                    .advice-card {
                        padding: 24px;
                        border-radius: 12px;
                        margin: 12px 0;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                        transition: transform 0.2s ease-in-out;
                    }
                    
                    .advice-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
                    }
                    
                    .advice-title {
                        font-size: 18px;
                        font-weight: 600;
                        margin-bottom: 12px;
                        color: white;
                        opacity: 0.95;
                    }
                    
                    .advice-content {
                        font-size: 16px;
                        line-height: 1.6;
                        color: white;
                        opacity: 0.9;
                    }
                    
                    .advice-icon {
                        font-size: 24px;
                        margin-right: 12px;
                        vertical-align: middle;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Create two rows with two columns each
            for i in range(0, len(advice_list), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(advice_list):
                        advice = advice_list[i + j]
                        with cols[j]:
                            st.markdown(f"""
                                <div class="advice-card" style="background-color: {advice['color']};">
                                    <div class="advice-title">
                                        <span class="advice-icon">{advice['icon']}</span>
                                        {advice['title']}
                                    </div>
                                    <div class="advice-content">
                                        {advice['advice']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

    except Exception as e:
        print(f"Error: {e}")
        st.error("Unable to generate advice at this time. Please try again.")

def display_progress_sidebar(feature_order):
    """Display progress of features visited in sidebar"""
    if 'visited_features' in st.session_state:
        st.sidebar.markdown("### Your Progress")
        for feature in feature_order.keys():
            feature_key = feature.lower().replace(" ", "_")
            if feature_key in st.session_state.visited_features or feature in st.session_state.visited_features:
                st.sidebar.markdown(f"✅ {feature}")
            else:
                st.sidebar.markdown(f"◽ {feature}")

def display_conversation_starters(situation):
    st.markdown("<h2 class='section-header'>Conversation Starters</h2>", unsafe_allow_html=True)
    
    if not situation:
        st.warning("Please describe the situation to get conversation starters.")
        return

    try:
        messages = [
            {"role": "system", "content": f"""
                Based on these citations:
                {CONVERSATION_STARTER_CITATIONS}
                {WEBSITE_CITATIONS}
                
                Generate 5 complete, clear conversation starters for a parent talking to their child about {situation}
                Each starter should be a full question or statement.
                Focus on open-ended, empathetic approaches that encourage dialogue.
                
                Format each starter as a JSON object with:
                1. The conversation starter text
                2. A category from: "Feelings", "Understanding", "Activities", "Solutions", "Ideas"
                3. A number (1-5)
                
                Return as a JSON array of objects with properties:
                - text: the complete conversation starter
                - category: category name
                - number: starter number (1-5)
                
                Ensure each starter:
                - Is open-ended and encourages dialogue
                - Shows empathy and understanding
                - Is age-appropriate
                - Addresses the specific situation
                - Uses clear, simple language
            """},
            {"role": "user", "content": f"Generate conversation starters for this situation: {situation}"}
        ]
        
        with st.spinner("Generating conversation starters..."):
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            starters = json.loads(completion.choices[0].message.content)
            
            # Add custom CSS
            st.markdown("""
                <style>
                    .starter-card {
                        background-color: white;
                        border-radius: 12px;
                        padding: 24px;
                        margin-bottom: 20px;
                        border: 1px solid #e2e8f0;
                        transition: transform 0.2s, box-shadow 0.2s;
                    }
                    
                    .starter-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
                    }
                    
                    .category-badge {
                        display: inline-block;
                        padding: 6px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        font-weight: 600;
                        margin-bottom: 12px;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    }
                    
                    .number-badge {
                        position: absolute;
                        top: -10px;
                        left: -10px;
                        width: 28px;
                        height: 28px;
                        border-radius: 50%;
                        background-color: #4F46E5;
                        color: white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 600;
                        font-size: 14px;
                        border: 2px solid white;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    }
                    
                    .starter-text {
                        font-size: 16px;
                        line-height: 1.6;
                        color: #2d3748;
                    }
                    
                    .tips-section {
                        background-color: #f7fafc;
                        border-radius: 12px;
                        padding: 24px;
                        margin-top: 40px;
                        border: 1px solid #e2e8f0;
                    }
                    
                    .tips-header {
                        display: flex;
                        align-items: center;
                        margin-bottom: 16px;
                    }
                    
                    .tips-icon {
                        width: 32px;
                        height: 32px;
                        margin-right: 12px;
                        background-color: #4F46E5;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Color mapping for categories
            color_map = {
                "Feelings": "#EBF4FF:#4299E1",      # Blue
                "Understanding": "#FEF3F2:#F98080",  # Red
                "Activities": "#F0FDF4:#34D399",     # Green
                "Solutions": "#FDF2F8:#EC4899",      # Pink
                "Ideas": "#FDF6B2:#D97706"          # Yellow
            }
            
            # Display introduction
            st.markdown("""
                <div style="margin-bottom: 24px;">
                    <p style="color: #4a5568; font-size: 16px; line-height: 1.6;">
                        Use these thoughtfully crafted conversation starters to open up meaningful dialogue with your child. 
                        Each starter is designed to encourage open communication and understanding.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display starters in a single column for better readability
            for starter in starters:
                bg_color, text_color = color_map.get(starter['category'], "#F9FAFB:#4A5568").split(":")
                
                st.markdown(f"""
                    <div class="starter-card" style="position: relative;">
                        <div class="number-badge">{starter['number']}</div>
                        <div style="display: flex; align-items: flex-start;">
                            <div style="flex: 1;">
                                <div class="category-badge" style="background-color: {bg_color}; color: {text_color};">
                                    {starter['category']}
                                </div>
                                <div class="starter-text">
                                    "{starter['text']}"
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Display tips section
            st.markdown("""
                <div class="tips-section">
                    <div class="tips-header">
                        <div class="tips-icon">💡</div>
                        <h3 style="color: #2d3748; font-size: 20px; font-weight: 600; margin: 0;">
                            Tips for Using These Starters
                        </h3>
                    </div>
                    <ul style="color: #4a5568; list-style-type: none; padding: 0; margin: 0;">
                        <li style="margin-bottom: 12px; display: flex; align-items: center;">
                            <span style="width: 8px; height: 8px; background-color: #4F46E5; border-radius: 50%; margin-right: 12px;"></span>
                            Choose a calm moment when both you and your child are ready to talk
                        </li>
                        <li style="margin-bottom: 12px; display: flex; align-items: center;">
                            <span style="width: 8px; height: 8px; background-color: #4F46E5; border-radius: 50%; margin-right: 12px;"></span>
                            Use a gentle, curious tone that shows you're interested in their perspective
                        </li>
                        <li style="margin-bottom: 12px; display: flex; align-items: center;">
                            <span style="width: 8px; height: 8px; background-color: #4F46E5; border-radius: 50%; margin-right: 12px;"></span>
                            Give your child time to process and respond without rushing
                        </li>
                        <li style="display: flex; align-items: center;">
                            <span style="width: 8px; height: 8px; background-color: #4F46E5; border-radius: 50%; margin-right: 12px;"></span>
                            Listen actively and validate their feelings to create an environment where they feel comfortable expressing themselves
                        </li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        print(f"Error generating conversation starters: {e}")
        st.error("Unable to generate conversation starters at this time.")

def display_communication_techniques(situation):
    st.markdown("<h2 class='section-header'>Communication Techniques</h2>", unsafe_allow_html=True)
    
    if not situation:
        st.warning("Please describe the situation to get communication techniques.")
        return

    try:
        with st.spinner('Generating techniques...'):
            messages = [
                {"role": "system", "content": f"""
                    You are a parenting expert. Generate 3 communication strategies for the given situation.
                    Each strategy should include:
                    1. A clear name and emoji icon
                    2. A single-sentence purpose
                    3. Three specific action steps
                    4. A brief, realistic example
                    
                    Format as markdown with this exact structure:
                    ### [emoji] Strategy Name
                    
                    Purpose: [one clear sentence]
                    
                    A. [first step]
                    B. [second step]
                    C. [third step]
                    
                    Example: [brief specific example under 25 words]
                    
                    Make each strategy practical and specific to the situation.
                    Base advice on child development research and effective communication principles.
                    {COMMUNICATION_STRATEGIES_CITATIONS}
                """},
                {"role": "user", "content": f"Generate parenting communication strategies for this situation: {situation}"}
            ]
            
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )

            strategies = completion.choices[0].message.content.split('###')[1:]  # Split by ### and remove empty first element
            
            # Add custom CSS
            st.markdown("""
                <style>
                    .strategy-card {
                        background-color: white;
                        border-radius: 12px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin-bottom: 20px;
                        overflow: hidden;
                    }
                    
                    .card-header {
                        padding: 16px;
                        font-size: 1.2em;
                        font-weight: 600;
                        color: white;
                    }
                    
                    .card-content {
                        padding: 20px;
                    }
                    
                    .purpose {
                        background-color: #f8fafc;
                        padding: 12px;
                        border-radius: 8px;
                        margin-bottom: 16px;
                    }
                    
                    .step {
                        padding: 12px;
                        margin-bottom: 8px;
                        border: 1px solid #e5e7eb;
                        border-radius: 6px;
                    }
                    
                    .example {
                        background-color: #f8fafc;
                        padding: 16px;
                        border-radius: 8px;
                        font-style: italic;
                        margin-top: 16px;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Create three columns
            cols = st.columns(3)
            colors = ["#7C3AED", "#0D9488", "#D97706"]  # violet-600, teal-600, amber-600
            
            for idx, (strategy, col, color) in enumerate(zip(strategies, cols, colors)):
                with col:
                    # Parse strategy content
                    lines = strategy.strip().split('\n')
                    title = lines[0].strip()
                    purpose = next(line for line in lines if line.startswith('Purpose:')).replace('Purpose:', '').strip()
                    steps = [line.strip() for line in lines if line.strip().startswith(('A.', 'B.', 'C.'))]
                    example = next(line for line in lines if line.startswith('Example:')).replace('Example:', '').strip()
                    
                    st.markdown(f"""
                        <div class="strategy-card">
                            <div class="card-header" style="background-color: {color}">
                                {title}
                            </div>
                            <div class="card-content">
                                <div class="purpose">
                                    {purpose}
                                </div>
                                <div class="steps">
                                    <div class="step">
                                        {steps[0]}
                                    </div>
                                    <div class="step">
                                        {steps[1]}
                                    </div>
                                    <div class="step">
                                        {steps[2]} 
                                    </div>
                                </div>
                                <div class="example">
                                    <strong>✨ Example:</strong><br>
                                    {example}
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        traceback.print_exc()  # This will print the full error traceback to your logs
        st.error("Unable to generate communication techniques. Please try again.")

def simulate_conversation_streamlit(name: str, child_age: str, situation: str):
    # Custom CSS to style the expander
    st.markdown("""
    <style>
    /* Custom styling for selectbox */
    .stSelectbox [data-baseweb=select] {
        background-color: white;
        border: 2px solid #4338CA !important;  /* Thicker border with indigo color */
        border-radius: 8px !important;
        padding: 6px 12px;
    }
    
    .stSelectbox [data-baseweb=select]:hover {
        border-color: #6366F1 !important;  /* Lighter indigo on hover */
        box-shadow: 0 2px 4px rgba(99, 102, 241, 0.1);
    }
    
    /* Style for the dropdown arrow */
    .stSelectbox [data-baseweb=select] svg {
        color: #4338CA;  /* Indigo color for the dropdown arrow */
    }
    
    /* Style for the selectbox label */
    .stSelectbox > label {
        color: #4338CA !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)
    
    # Add header above the expander
    st.markdown("### Getting Started")
    
    with st.expander("Click here for instructions"):
        st.write("The simulation will provide real-time feedback on your responses and generate age-appropriate replies based on your chosen communication strategy.")
        
        instructions = [
            {
                "step": "1",
                "title": "Select your communication strategy below",
                "desc": "Choose from Active Listening, Positive Reinforcement, or Reflective Questioning"
            },
            {
                "step": "2",
                "title": "Type your responses in the text area",
                "desc": "Consider your chosen strategy when formulating your response"
            },
            {
                "step": "3",
                "title": "Click 'Send Response' to continue the conversation",
                "desc": "You'll receive feedback and see how your child responds"
            },
            {
                "step": "4",
                "title": "Click 'End Conversation' when you're ready to finish",
                "desc": "You'll see a review of your conversation with detailed feedback"
            }
        ]
        
        for instruction in instructions:
            st.markdown(f"""
                <div style='display: flex; gap: 1rem; margin-bottom: 1rem;'>
                    <div style='background-color: #e8eeff; color: #3b82f6; width: 28px; height: 28px; 
                         border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                         font-weight: 600; flex-shrink: 0;'>
                        {instruction['step']}
                    </div>
                    <div>
                        <div style='font-weight: 500; color: #1f2937;'>{instruction['title']}</div>
                        <div style='color: #6b7280; font-size: 0.875rem;'>{instruction['desc']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("### Communication Strategy")
    
    with st.expander("Select your communication strategy and learn more about each approach"):
        strategy_descriptions = {
            "Active Listening": """
                👂 **Active Listening**
                
                Fully focus on, understand, and remember what your child is saying. This helps them feel heard and valued.
                
                Key aspects:
                - Give full attention when your child speaks
                - Show you're listening through body language
                - Reflect back what you hear to confirm understanding
                - Avoid interrupting or jumping to conclusions
            """,
            "Positive Reinforcement": """
                ⭐ **Positive Reinforcement**
                
                Encourage desired behaviors through specific praise or rewards, helping build self-esteem and motivation.
                
                Key aspects:
                - Praise specific actions rather than general behavior
                - Focus on effort and improvement
                - Use encouraging words and genuine appreciation
                - Recognize progress, not just perfect results
            """,
            "Reflective Questioning": """
                ❓ **Reflective Questioning**
                
                Use open-ended questions to help children think deeper and express themselves.
                
                Key aspects:
                - Ask questions that can't be answered with yes/no
                - Help children explore their feelings and thoughts
                - Show genuine curiosity about their perspective
                - Give time to think and respond
            """
        }
        
        strategy = st.selectbox(
            "Choose your approach:",
            ["Active Listening", "Positive Reinforcement", "Reflective Questioning"],
            key=f"strategy_select_{st.session_state['simulation_id']}"
        )
        
        st.markdown(strategy_descriptions[strategy])
    
    st.session_state['strategy'] = strategy

    # Single Conversation section
    st.markdown("### Conversation")
    st.info(f"Current Situation: {situation}")
    
    # Chat history display
    if st.session_state['conversation_history']:
        for msg in st.session_state['conversation_history']:
            speaker = "You" if msg['role'] == 'parent' else st.session_state.get('child_name', 'Child')
            message_class = 'message-parent' if msg['role'] == 'parent' else 'message-child'
            
            cols = st.columns([8, 4])
            with cols[0]:
                st.markdown(f"""
                    <div class='{message_class}'>
                        <strong>{speaker}:</strong> {msg['content']}
                    </div>
                """, unsafe_allow_html=True)
            
            if msg['role'] == 'parent' and 'feedback' in msg:
                with cols[1]:
                    st.info(f"💡 {msg['feedback']['hint']}")

    # Input area
    user_input = st.text_area(
        label="Your response",
        placeholder="How would you start this conversation with your child? Type here...",
        key=f"parent_input_{st.session_state['simulation_id']}_{st.session_state['turn_count']}",
        height=100
    )
    
    # Action buttons
    send_cols = st.columns(2)
    with send_cols[0]:
        send_button = st.button("Send Response", use_container_width=True)
    with send_cols[1]:
        end_button = st.button("End Conversation", use_container_width=True, type="secondary")
    
    # Handle input
    if send_button or end_button:
        handle_conversation_input(send_button, end_button, user_input, child_age, situation)

def handle_user_input(child_age: str, situation: str):
    """Handle user input during conversation"""
    user_input = st.text_area(
        label="Your response",
        placeholder="How would you respond to your child? Type here...",
        key=f"parent_input_{st.session_state['simulation_id']}_{st.session_state['turn_count']}",
        height=100
    )
    
    submit_cols = st.columns(2)
    with submit_cols[0]:
        send_button = st.button("Send Response", key="send_response", use_container_width=True)
    with submit_cols[1]:
        end_button = st.button("End Conversation", key="end_conversation", use_container_width=True, type="secondary")
    
    handle_conversation_input(send_button, end_button, user_input, child_age, situation)

def handle_conversation_input(send_button: bool, end_button: bool, user_input: str, child_age: str, situation: str):
    if send_button and user_input:
        feedback = provide_realtime_feedback(
            user_input, 
            st.session_state['strategy'],
            st.session_state['situation'],
            child_age,
            st.session_state['conversation_history']
        )
        
        simulation_data = {
            "user_id": st.session_state['parent_name'],
            "simulation_data": {
                "parent_message": user_input,
                "strategy": st.session_state['strategy'],
                "child_age": child_age,
                "situation": situation,
                "turn_count": st.session_state['turn_count']
            },
            "created_at": datetime.utcnow().isoformat()
        }
        
        success, result = supabase_manager.save_simulation_data(simulation_data)
        if not success:
            st.error(f"Failed to save simulation data: {result}")
            return
            
        if isinstance(result, str):
            st.session_state['current_simulation_id'] = result
        
        # Add parent's message to history
        st.session_state['conversation_history'].append({
            "role": "parent",
            "content": user_input,
            "id": len(st.session_state['conversation_history']),
            "feedback": feedback,
            "strategy_used": st.session_state['strategy']
        })

        update_langsmith_run(
            st.session_state['run_id'],
            {
                f"turn_{st.session_state['turn_count']}_parent": {
                    "content": user_input,
                    "strategy": st.session_state['strategy'],
                    "feedback": feedback
                }
            }
        )
    
        # Generate child's response
        child_response = generate_child_response(
            st.session_state['conversation_history'],
            child_age,
            situation,
            st.session_state['child_mood'],
            st.session_state['strategy'],
            user_input
        )
        
        # Add child's response to history
        st.session_state['conversation_history'].append({
            "role": "child",
            "content": child_response,
            "id": len(st.session_state['conversation_history'])
        })
        
        st.session_state['turn_count'] += 1
        st.rerun()
    
    if end_button:
        update_langsmith_run(
            st.session_state['run_id'],
            {
                "conversation_ended": True,
                "total_turns": st.session_state['turn_count']
            }
        )
        end_simulation(st.session_state['conversation_history'], child_age, st.session_state['strategy'])

def display_conversation_playback(conversation_history):
    st.markdown("<h2 class='section-header'>Conversation Review</h2>", unsafe_allow_html=True)
    
    if not conversation_history:
        st.info("No conversation history to display.")
        return

    for msg in conversation_history:
        speaker = "You" if msg['role'] == "parent" else st.session_state.get('child_name', 'Child')
        background = "#f0f7ff" if msg["role"] == "parent" else "#f5f5f5"
        
        st.markdown(f"""
            <div style='padding: 15px; border-radius: 8px; margin: 10px 0; background-color: {background};'>
                <div><strong>{speaker}:</strong> {msg['content']}</div>
                {f"<div style='margin-top: 8px; color: #666;'><i>Strategy used: {msg['strategy_used']}</i></div>" if msg.get('strategy_used') else ""}
                {f"<div style='margin-top: 8px; color: #666;'><i>💡 {msg['feedback']['detailed']}</i></div>" if msg.get('feedback') else ""}
            </div>
        """, unsafe_allow_html=True)

def end_simulation(conversation_history: list, child_age: str, strategy: str):
    st.session_state['simulation_ended'] = True
    
    if not st.session_state.get('parent_name'):
        st.error("Error: Prolific ID not found. Please ensure you've entered your ID correctly.")
        return
    
    current_simulation_id = st.session_state.get('current_simulation_id')
    if current_simulation_id:
        success, message = supabase_manager.complete_simulation(
            current_simulation_id,
            st.session_state.get('run_id')
        )
        if not success:
            st.error(f"Failed to complete simulation: {message}")
    
    track_feature_visit("Role-Play Simulation")
    st.write("The simulation has ended.")
    
    display_conversation_playback(conversation_history)
    
    # Check visited features and display completion message
    required_features = {'advice', 'communication_techniques', 'conversation_starters', 'role_play_simulation'}
    visited = {f.lower().replace(" ", "_").replace("-", "_") for f in st.session_state.get('visited_features', [])}
    
    if required_features.issubset(visited):
        st.markdown("""
        <div style='background-color: #f0fff4; padding: 20px; border-radius: 8px; border: 1px solid #68d391; margin-top: 20px;'>
            <h3 style='color: #2f855a; margin-bottom: 10px;'>🎉 Session Complete!</h3>
            <p>You've successfully explored all features of the Parenting Support Bot</p>
            <ul style='margin: 10px 0;'>
                <li>✓ Expert Advice</li>
                <li>✓ Communication Techniques</li>
                <li>✓ Conversation Starters</li>
                <li>✓ Role-Play Simulation</li>
            </ul>
            <p>Feel free to try another situation or explore different strategies by clicking "Edit Information" to start fresh.</p>
        </div>
        """, unsafe_allow_html=True)

def show_info_screen():
    """Display the initial information collection screen"""
    st.markdown("<h1 class='main-header'>Welcome to Parenting Support Bot</h1>", unsafe_allow_html=True)
    
    with st.form(key='parent_info_form'):
        st.markdown("<h2 class='section-header'>Please Tell Us About You</h2>", unsafe_allow_html=True)
        
        st.markdown("""
            <p class='description-text'>
                Please enter your <b>Prolific ID</b> (24-character identifier from your Prolific account)
            </p>
        """, unsafe_allow_html=True)
        
        parent_name = st.text_input("Prolific ID", 
                                  placeholder="Enter your 24-character Prolific ID...",
                                  help="This is the ID assigned to you by Prolific")
        
        child_name = st.text_input("Child's Name", 
                                 placeholder="Enter your child's name...")
        
        child_age = st.selectbox("Child's Age Range",
                               ["3-5 years", "6-9 years", "10-12 years"])
        
        situation = st.text_area("Situation Description",
                               placeholder="Type your situation here...",
                               height=120)
        
        submit_button = st.form_submit_button("Start", use_container_width=True)
        
        if submit_button:
            if not parent_name or not child_name or not situation:
                st.error("Please fill in all fields")
            elif len(parent_name) != 24:
                st.error("Please enter a valid 24-character Prolific ID")
            else:
                st.session_state['parent_name'] = parent_name
                st.session_state['child_name'] = child_name
                st.session_state['child_age'] = child_age
                st.session_state['situation'] = situation
                st.session_state['info_submitted'] = True
                st.rerun()

def show_tutorial():
    """Display tutorial for first-time users"""
    st.markdown("# Welcome to the Parenting Support Bot! 🎉")
    
    st.markdown("""
        This app is designed to help you develop effective parenting strategies through:
        
        - 📚 **Expert Advice** - Get evidence-based parenting advice
        - 🗣️ **Communication Techniques** - Discover effective communication strategies
        - 💭 **Conversation Starters** - Learn how to begin difficult conversations
        - 🎮 **Role-Play Simulation** - Practice conversations in a safe environment
    """)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Got it, let's start!", use_container_width=True):
            st.session_state.show_tutorial = False
            st.rerun()

def reset_simulation():
    st.session_state['conversation_history'] = []
    st.session_state['turn_count'] = 0
    st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
    st.session_state['simulation_id'] = str(uuid4())

# Initialize Supabase manager
supabase_manager = SupabaseManager()

def main():
    # Test Supabase connection first
    def test_supabase_connection():
        """Test Supabase connection and configuration"""
        print("\n=== Testing Supabase Connection ===")
        
        # Check environment variables
        if hasattr(st, 'secrets'):
            print("Using Streamlit secrets")
            has_url = bool(st.secrets.get('SUPABASE_URL'))
            has_key = bool(st.secrets.get('SUPABASE_KEY'))
        else:
            print("Using environment variables")
            has_url = bool(os.getenv('SUPABASE_URL'))
            has_key = bool(os.getenv('SUPABASE_KEY'))
        
        print(f"Has Supabase URL: {has_url}")
        print(f"Has Supabase Key: {has_key}")
        
        # Test connection
        try:
            manager = SupabaseManager()
            success = manager.initialize()
            if success:
                print("✅ Supabase connection successful")
                # Test table access
                result = manager.supabase.table('simulations').select("count", count='exact').limit(1).execute()
                print("✅ Simulations table accessible")
                return True
            else:
                print("❌ Failed to initialize Supabase")
                return False
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            traceback.print_exc()
            return False

    # Run connection test before proceeding
    if not test_supabase_connection():
        st.error("Failed to connect to database. Please check configuration.")
        st.stop()

    inject_qualtrics_messenger()

    params = st.experimental_get_query_params()
    
    if 'prolific_id' in params:
        st.session_state['parent_name'] = params['prolific_id'][0]
    if 'child_name' in params:
        st.session_state['child_name'] = params['child_name'][0]
    if 'child_age' in params:
        st.session_state['child_age'] = params['child_age'][0]
    if 'feature' in params:
        selected_feature = params['feature'][0]
    if 'situation' in params:
        st.session_state['situation'] = params['situation'][0]
    if 'feature' in params:
        selected_feature = params['feature'][0]
    else:
        selected_feature = "Advice"
        
    # If we have all required parameters, mark as submitted
    if all(key in st.session_state for key in ['parent_name', 'child_name', 'child_age', 'situation']):
        st.session_state['info_submitted'] = True

    # Initialize feature order and descriptions
    feature_order = {
        "Advice": "Get expert guidance on handling specific parenting situations based on evidence-based strategies.",
        "Communication Techniques": "Discover helpful ways to communicate with your child and get tips on how to address your specific situation.",
        "Conversation Starters": "Receive help initiating difficult conversations with suggested opening phrases and questions.",
        "Role-Play Simulation": "Practice conversations in a safe environment to develop and refine your communication approach."
    }
         
    if not st.session_state.get('info_submitted', False):
        show_info_screen()
        return

    with st.sidebar:
        st.markdown("<h3 class='subsection-header'>Current Information</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='info-section'>
                <strong>Parent:</strong> {st.session_state['parent_name']}<br>
                <strong>Child:</strong> {st.session_state['child_name']}<br>
                <strong>Age:</strong> {st.session_state['child_age']}<br>
                <strong>Situation:</strong> {st.session_state['situation']}
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Edit Information", use_container_width=True):
            st.session_state['info_submitted'] = False
            st.session_state.pop('conversation_history', None)
            st.session_state.pop('run_id', None)
            st.rerun()

    if 'show_tutorial' not in st.session_state:
        show_tutorial()
    else:
        st.markdown("<h1 class='main-header'>Parenting Support Bot</h1>", unsafe_allow_html=True)

        selected = st.radio(
            "Choose an option:",
            list(feature_order.keys()),
            horizontal=True,
            help="Select a tool that best matches your current needs"
        )

        st.info(feature_order[selected])

     # Track and display selected feature
        if selected == "Advice":
            track_feature_visit("advice")
            display_advice(st.session_state['parent_name'], st.session_state['child_age'], st.session_state['situation'])
        elif selected == "Communication Techniques":
            track_feature_visit("communication_techniques")
            display_communication_techniques(st.session_state['situation'])
        elif selected == "Conversation Starters":
            track_feature_visit("conversation_starters")
            display_conversation_starters(st.session_state['situation'])
        elif selected == "Role-Play Simulation":
            track_feature_visit("role_play")
            simulate_conversation_streamlit(st.session_state['parent_name'], st.session_state['child_age'], st.session_state['situation'])

        display_progress_sidebar(feature_order)
    
if __name__ == "__main__":
    try:
        init_session_state()
        check_environment()
        
        if not supabase_manager.initialize():
            st.error("Failed to initialize Supabase connection!")
            st.stop()
        
        main()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        print(f"Detailed error: {str(e)}")