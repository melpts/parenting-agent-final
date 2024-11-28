import streamlit as st
import os
import openai
from dotenv import load_dotenv
import json
import warnings
from datetime import datetime
import random
from uuid import UUID, uuid4
from typing import Optional, Dict, Any

# Set page config as the first Streamlit command
st.set_page_config(
    layout="wide", 
    page_title="Parenting Support Bot",
    initial_sidebar_state="expanded"
)

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Supabase import with error handling
try:
    from supabase import create_client, Client
except ImportError:
    st.error("""
        Unable to import Supabase client. Please ensure the package is installed:
        ```
        pip install supabase
        ```
        If you're using Streamlit Cloud, verify that 'supabase' is in your requirements.txt file.
    """)
    st.stop()

# Langchain imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import StringPromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import HumanMessage, AIMessage, AgentAction, AgentFinish
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.manager import get_openai_callback


# Additional third-party imports
from streamlit_feedback import streamlit_feedback
from langsmith import Client
from langsmith.run_helpers import traceable
from langsmith.run_helpers import get_current_run_tree

# Citation imports
from conversation_starter_citations import CONVERSATION_STARTER_CITATIONS
from communication_strategies_citations import COMMUNICATION_STRATEGIES_CITATIONS
from simulation_citations import SIMULATION_CITATIONS
from Website_citations import WEBSITE_CITATIONS
from Active_listening_citations import ACTIVE_LISTENING_CITATIONS
from i_messages_citations import I_MESSAGES_CITATIONS
from positive_reinforcement import POSITIVE_REINFORCEMENT_CITATIONS
from Reflective_questioning import REFLECTIVE_QUESTIONING_CITATIONS

# Initialize LangSmith Client
smith_client = Client()

# Custom CSS
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

# Strategy Hints
STRATEGY_HINTS = {
    "Active Listening": [
        "- Repeat back what your child says to show understanding",
        "- Use phrases like 'I hear that you...' or 'It sounds like...'",
        "- Notice and name emotions: 'You seem frustrated'",
        "- Give your child full attention, maintain eye contact",
        "- Avoid interrupting or finishing their sentences",
        "- Acknowledge their perspective before sharing yours"
    ],
    "Positive Reinforcement": [
        "- Be specific about what behavior you're praising",
        "- Focus on effort rather than outcome",
        "- Use a warm, enthusiastic tone",
        "- Catch them being good and acknowledge it immediately",
        "- Describe the positive impact of their behavior",
        "- Use 'I notice' statements when praising"
    ],
    "Reflective Questioning": [
        "- Ask 'what' and 'how' questions instead of 'why'",
        "- Use open-ended questions to encourage sharing",
        "- Follow up their answers with curious questions",
        "- Avoid leading questions or suggesting answers",
        "- Show genuine interest in their perspective",
        "- Give them time to think and respond"
    ]
}

# Age-specific responses
AGE_SPECIFIC_RESPONSES = {
    "3-5 years": {
        "cooperative": [
            "Okay, I'll try...",
            "Can you help me?",
            "I want to be good!",
            "Like this, Mommy/Daddy?",
            "It's hard!",
            "I can try that!",
            "*showing effort* Is this better?",
            "*smiling* Can you help me?",
            "*wiping tears* I'm sorry",
            "*picking up toys* I can clean up!"
        ],
        "defiant": [
            "No! No! NO!",
            "*throwing self on floor* I DON'T WANNA!",
            "*covering ears* La la la, can't hear you!",
            "You're not the boss of me!",
            "I want MY way!",
            "NO NO NO!",
            "*turning away* Not listening!",
            "Don't want to!",
            "*flopping on floor* It's not FAIR!",
            "*pushing away* Leave me alone!"
        ],
        "distracted": [
            "Look, my toy is dancing!",
            "Can I have a snack?",
            "*spinning around* Wheeeee!",
            "But I wanna play with my blocks!",
            "Is it time for cartoons?",
            "But my show is on...",
            "*spinning in circles* Wheeee!",
            "*building with blocks* Just one more tower?",
            "*drawing* Need to finish coloring!",
            "But I'm not done yet!"
        ]
    },
    "6-9 years": {
        "cooperative": [
            "I'll clean up after I finish this part.",
            "Sorry, I didn't mean to...",
            "I promise I'll do better.",
            "Will you show me how?",
            "*putting down game* Okay, I understand",
            "I know I should... *starting task*",
            "*organizing things* I'm helping!",
            "You're right... *beginning task*",
            "*showing work* Is this better?"
        ],
        "defiant": [
            "But that's not fair! Jamie never has to!",
            "*arms crossed* You can't make me!",
            "I hate these rules!",
            "You never let me do anything fun!",
            "Well, Sarah's parents let her!",
            "But Emma's mom lets her!",
            "*rolling eyes* This isn't fair!",
            "You NEVER let me do anything fun!",
            "*slamming door* Leave me alone!",
            "Why do I always have to?",
            "*arms crossed* Make me!",
            "You're the worst! *storming off*"
        ],
        "distracted": [
            "But first can I just...",
            "Wait, I forgot to tell you about...",
            "Can we do it later? I'm almost finished with...",
            "Oh! I just remembered something!",
            "*playing game* Just need to save...",
            "But first can I...",
            "Did you know that... *changing subject*",
            "*watching YouTube* Almost done!",
            "Wait, I forgot to tell you about...",
            "*fixated on phone* In a minute..."
        ]
    },
    "10-12 years": {
        "cooperative": [
            "Fine, I get it. Just give me a minute.",
            "I know, I know. I'm going.",
            "Okay, but can we talk about it first?",
            "I understand, but...",
            "I'll do it, just let me finish this.",
            "*putting phone down* Okay, I'm listening",
            "Can we talk about it?",
            "*nodding* That's fair",
            "I understand... *complying*",
            "*showing compromise* How about this?"
        ],
        "defiant": [
            "This is so unfair! You never understand!",
            "Everyone else gets to!",
            "*slamming door* Leave me alone!",
            "You're ruining everything!",
            "*eye roll* Whatever",
            "This is SO unfair! *texting friends*",
            "You don't understand ANYTHING!",
            "*slamming door* I hate this!",
            "Everyone else's parents...",
            "*storming off* You're ruining my life!",
            "This is stupid! *throwing phone*"
        ],
        "distracted": [
            "Yeah, just one more level...",
            "Hold on, I'm texting...",
            "In a minute... I'm doing something.",
            "But I'm in the middle of something!",
            "*texting* Just one sec...",
            "But my friends are waiting online...",
            "*watching TikTok* Almost done",
            "*gaming* Can't pause multiplayer!",
            "Hold on, I'm in the middle of...",
            "*scrolling phone* Five more minutes?"
        ]
    }
}

# Apply custom CSS
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
        
    def initialize(self):
        try:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            return True
        except Exception as e:
            print(f"Error initializing Supabase: {e}")
            return False
    
    def save_reflection(self, user_id: str, reflection_type: str, content: Dict[str, Any]) -> bool:
        try:
            data = {
                "user_id": user_id,
                "type": reflection_type, 
                "content": content,
                "created_at": datetime.utcnow().isoformat(),
                "langsmith_run_id": st.session_state.get('run_id')
            }
            print("Attempting to save reflection:", data)
            result = self.supabase.table('reflections').insert(data).execute()
            print("Save result:", result)
            return True if result.data else False
        except Exception as e:
            print(f"Error saving reflection: {e}")
            return False
    
    def get_reflections(self, user_id: str) -> list:
        try:
            result = self.supabase.table('reflections')\
                .select("*")\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .execute()
            return result.data if result else []
        except Exception as e:
            print(f"Error retrieving reflections: {e}")
            return []
    
    def save_simulation_data(self, simulation_data: Dict[str, Any]) -> bool:
        try:
            print("Saving simulation data:", simulation_data)
            result = self.supabase.table('simulations').insert(simulation_data).execute()
            print("Save result:", result)
            return True if result.data else False
        except Exception as e:
            print(f"Error saving simulation data: {e}")
            return False

    def view_all_simulations(self):
        try:
            result = self.supabase.table('simulations').select("*").execute()
            return result.data
        except Exception as e:
            print(f"Error fetching simulations: {e}")
            return []

    def view_user_simulations(self, user_id: str):
        try:
            result = self.supabase.table('simulations').select("*").eq('user_id', user_id).execute()
            print("Query result:", result)
            return result.data
        except Exception as e:
            print(f"Error fetching simulations: {e}")
            return []

    def save_simulation(self, simulation_data: Dict[str, Any]) -> bool:
        try:
            result = self.supabase.table('simulations').insert(simulation_data).execute()
            return True if result.data else False
        except Exception as e:
            print(f"Error saving simulation: {e}")
            return False

# Initialize Supabase manager
supabase_manager = SupabaseManager()

def display_stored_data():
    if st.session_state.get('parent_name'):
        st.markdown("### Stored Simulation Data")
        simulations = supabase_manager.view_user_simulations(st.session_state['parent_name'])
        if simulations:
            for sim in simulations:
                with st.expander(f"Simulation {sim['id']}"):
                    st.json(sim)
        else:
            st.info("No simulations found")

def setup_environment():
    """Initialize environment variables and connections"""
    load_dotenv()
    
    if hasattr(st, 'secrets'):
        # OpenAI setup
        openai.api_key = st.secrets.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = st.secrets.get('OPENAI_API_KEY', '')
        
        # LangChain setup
        os.environ['LANGCHAIN_API_KEY'] = st.secrets.get('LANGCHAIN_API_KEY', '')
        os.environ['LANGCHAIN_PROJECT'] = st.secrets.get('LANGCHAIN_PROJECT', 'Parenting agent2')
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
        
        # Initialize Supabase
        if not supabase_manager.initialize():
            st.error('Failed to initialize Supabase connection!')
            st.stop()
    else:
        st.error('Required secrets not found!')
        st.stop()

def check_environment():
    """Check all required environment variables and secrets"""
    required_secrets = [
        'OPENAI_API_KEY',
        'LANGCHAIN_API_KEY',
        'LANGCHAIN_PROJECT',
        'SUPABASE_URL',
        'SUPABASE_KEY',
        'ADMIN_PASSWORD'
    ]
    
    missing = []
    if hasattr(st, 'secrets'):
        # Debug: Print available secrets (keys only, not values)
        #st.write("Available secrets:", list(st.secrets.keys()))
        
        for secret in required_secrets:
            if not st.secrets.get(secret):
                missing.append(secret)
    else:
        missing = required_secrets
    
    if missing:
        st.error(f"Missing required secrets: {', '.join(missing)}")
        st.info("Please check your .streamlit/secrets.toml file or Streamlit Cloud settings.")
        st.stop()

def setup_memory():
    """Initialize conversation memory"""
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(
        chat_memory=msgs,
        return_messages=True,
        memory_key="chat_history"
    )
    return memory

# Initialize memory
memory = setup_memory()

def init_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'run_id': str(uuid4()),
        'agentState': "start",
        'consent': False,
        'exp_data': True,
        'llm_model': "gpt-4",
        'simulation_ended': False,
        'stored_responses': {},
        'show_hints': False,
        'paused': False,
        'info_submitted': False,
        'conversation_history': [],
        'child_mood': random.choice(['cooperative', 'defiant', 'distracted']),
        'turn_count': 0,
        'strategy': "Active Listening",
        'simulation_id': str(uuid4()),
        'visited_features': set()
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def track_feature_visit(feature_name: str):
    """Track which features the user has visited"""
    if 'visited_features' not in st.session_state:
        st.session_state.visited_features = set()
    
    feature_display_map = {
        "advice": "Advice",
        "conversation_starters": "Conversation Starters",
        "communication_techniques": "Communication Techniques",
        "role_play": "Role-Play Simulation",
        "reflections": "View Reflections",
        "Role-Play Simulation": "Role-Play Simulation",
        "View Reflections": "View Reflections"
    }
    
    normalized_name = feature_display_map.get(feature_name, feature_name)
    st.session_state.visited_features.add(normalized_name)
    st.session_state.visited_features.add(normalized_name.lower().replace(" ", "_"))

def create_langsmith_run(name: str, inputs: Dict[str, Any], fallback_id: Optional[str] = None) -> str:
    """Create a new LangSmith run"""
    try:
        run = smith_client.create_run(
            run_type="chain",
            name=name,
            inputs=inputs
        )
        return str(run.id) if run else str(uuid4())
    except Exception as e:
        print(f"Error creating LangSmith run: {e}")
        return fallback_id or str(uuid4())

def update_langsmith_run(run_id: str, outputs: Dict[str, Any]):
    """Update an existing LangSmith run"""
    if not run_id:
        return
    try:
        try:
            UUID(run_id)
            smith_client.update_run(run_id, outputs=outputs)
        except ValueError:
            new_uuid = str(uuid4())
            print(f"Invalid UUID {run_id}, using new UUID: {new_uuid}")
            smith_client.update_run(new_uuid, outputs=outputs)
    except Exception as e:
        print(f"Error updating LangSmith run: {e}")

@traceable(name="generate_child_response")
def generate_child_response(conversation_history, child_age, situation, mood, strategy, parent_response):
    """Generate a child's response based on the current context"""
    child_name = st.session_state.get('child_name', 'the child')
    response_key = f"{parent_response}_{child_age}_{mood}_{strategy}"
    
    if response_key in st.session_state['stored_responses']:
        return st.session_state['stored_responses'][response_key]
    
    messages = [
        {"role": "system", "content": f"""You are {child_name}, a {child_age}-year-old child responding to your parent.
        Current mood: {mood}
        Situation: {situation}
        Recent context: {' | '.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])}

        Use these age-appropriate response patterns:
        {AGE_SPECIFIC_RESPONSES[child_age][mood]}

        IMPORTANT GUIDELINES:
        1. Always respond as {child_name} specifically
        2. Use age-appropriate language for {child_age}
        3. Show {mood} mood through words and actions
        4. Keep responses very short (1-2 sentences)
        5. Include emotional reactions (crying, stomping, etc.)
        6. Use simple vocabulary only
        7. Never explain rationally or use adult phrasing
        8. Reference siblings or friends when complaining about fairness
        9. Show raw emotion rather than logical thinking
        10. Sometimes include physical actions in *asterisks*
        """}
    ]

    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=60
        )
        response = completion.choices[0].message.content.strip()
        
        if 'run_id' in st.session_state:
            update_langsmith_run(
                st.session_state['run_id'],
                {
                    "parent_response": parent_response,
                    "child_response": response,
                    "strategy_used": strategy,
                    "mood": mood
                }
            )
        
        st.session_state['stored_responses'][response_key] = response
        return response
    except Exception as e:
        print(f"Error generating child response: {e}")
        return "I don't know what to say..."

def provide_realtime_feedback(parent_response: str, strategy: str) -> str:
    """Provide real-time feedback on parent's communication approach"""
    feedback_prompts = {
        "Active Listening": [
            "I noticed you reflected back what your child said. This shows you're listening.",
            "Try repeating key words your child used to show you're paying attention.",
            "Consider acknowledging your child's emotions in your response."
        ],
        "Positive Reinforcement": [
            "Great specific praise! Being specific helps reinforce the behavior.",
            "Try describing exactly what behavior you liked.",
            "Consider explaining why the behavior was helpful."
        ],
        "Reflective Questioning": [
            "Good open-ended question. This helps your child think deeper.",
            "Try asking 'what' or 'how' questions to encourage more detail.",
            "Consider following up on their answers with more questions."
        ]
    }
    
    strategy_feedback = random.choice(feedback_prompts[strategy])
    
    if "try" not in strategy_feedback.lower() and "consider" not in strategy_feedback.lower():
        suggestion = random.choice([
            f"\nTry using more {strategy.lower()} techniques in your next response.",
            f"\nConsider how you might incorporate more aspects of {strategy.lower()}.",
            f"\nThink about ways to deepen your use of {strategy.lower()}."
        ])
        strategy_feedback += suggestion
    
    return strategy_feedback

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
@traceable(name="display_advice")

def display_advice(parent_name: str, child_age: str, situation: str):
    """Display parenting advice based on the situation"""
    st.markdown("<h2 class='section-header'>Parenting Advice</h2>", unsafe_allow_html=True)
    if situation:
        try:
            with st.spinner('Processing your request...'):
                run_id = create_langsmith_run(
                    name="parenting_advice",
                    inputs={
                        "parent_name": parent_name,
                        "child_age": child_age,
                        "situation": situation
                    }
                )

                messages = [
                    {"role": "system", "content": "You are a parenting expert providing advice based on research-backed strategies."},
                    {"role": "user", "content": f"Parent: {parent_name}\nChild's age: {child_age}\nSituation: {situation}\nGoal: Get advice"}
                ]
                completion = openai.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )
                
                response = completion.choices[0].message.content
                update_langsmith_run(run_id, {"advice": response})
                st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please describe the situation in the sidebar to get advice.")

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
    """Display conversation starter suggestions"""
    st.markdown("<h2 class='section-header'>Conversation Starters</h2>", unsafe_allow_html=True)
    if not situation:
        st.warning("Please describe the situation to get conversation starters.")
        return

    try:
        with st.spinner('Generating conversation starters...'):
            run_id = create_langsmith_run(
                name="conversation_starters",
                inputs={"situation": situation}
            )
            
            messages = [
                {"role": "system", "content": f"""
                    Use these academic citations for guidance:
                    {CONVERSATION_STARTER_CITATIONS}
                    
                    And these website resources:
                    {WEBSITE_CITATIONS}
                    
                    Provide practical conversation starters for this situation.
                """},
                {"role": "user", "content": f"Situation: {situation}"}
            ]
            
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            starters = completion.choices[0].message.content
            update_langsmith_run(run_id, {"starters": starters})
            st.markdown(starters)
            
    except Exception as e:
        print(f"Error generating conversation starters: {e}")
        st.error("Unable to generate conversation starters at this time.")

def display_communication_techniques(situation):
    """Display communication technique suggestions"""
    st.markdown("<h2 class='section-header'>Communication Techniques</h2>", unsafe_allow_html=True)
    if not situation:
        st.warning("Please describe the situation to get communication techniques.")
        return

    try:
        with st.spinner('Generating communication techniques...'):
            run_id = create_langsmith_run(
                name="communication_techniques",
                inputs={"situation": situation}
            )
            
            messages = [
                {"role": "system", "content": f"""
                    Use these communication strategy citations:
                    {COMMUNICATION_STRATEGIES_CITATIONS}
                    
                    Provide specific, actionable communication techniques for this parenting situation.
                    Include examples and practical applications.
                """},
                {"role": "user", "content": f"Situation: {situation}"}
            ]
            
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            techniques = completion.choices[0].message.content
            update_langsmith_run(run_id, {"techniques": techniques})
            st.markdown(techniques)
            
    except Exception as e:
        print(f"Error generating communication techniques: {e}")
        st.error("Unable to generate communication techniques at this time.")

@traceable(name="simulate_conversation_streamlit")
def simulate_conversation_streamlit(name: str, child_age: str, situation: str):
    """Main conversation simulation interface"""
    name = st.session_state.get('parent_name', name)
    child_name = st.session_state.get('child_name', '')
    child_age = st.session_state.get('child_age', child_age)
    situation = st.session_state.get('situation', situation)

    st.markdown("## Parent-Child Role-Play Simulator")

    with st.container():
        st.markdown("""
            #### How to use this simulator:
            1. Start by responding naturally to your child's situation
            2. Try different communication strategies to see their impact
            3. Use the conversation controls to get hints or pause for reflection
        """)

    st.info(f"**Current Situation:** {situation}")

    # Strategy selection
    st.markdown("### Choose your communication strategy:")
    cols = st.columns(3)
    for i, (strategy, explanation) in enumerate(STRATEGY_EXPLANATIONS.items()):
        with cols[i]:
            if st.button(
                strategy,
                key=f"strategy_{strategy}_{st.session_state['simulation_id']}",
                use_container_width=True,
                type="primary" if strategy == st.session_state['strategy'] else "secondary"
            ):
                st.session_state['strategy'] = strategy
                update_langsmith_run(
                    st.session_state['run_id'],
                    {"strategy_change": strategy}
                )
                st.rerun()

    st.markdown(STRATEGY_EXPLANATIONS[st.session_state['strategy']], unsafe_allow_html=True)

    display_conversation_controls()
    display_conversation_history(child_name)
    handle_user_input(child_age, situation)

def display_conversation_controls():
    """Display conversation control buttons"""
    st.markdown("### Conversation Controls")
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("⏸️ Pause/Resume", use_container_width=True):
            st.session_state['paused'] = not st.session_state['paused']
    with action_cols[1]:
        if st.button("💡 Show Hints", use_container_width=True):
            st.session_state['show_hints'] = not st.session_state['show_hints']
    with action_cols[2]:
        if st.button("🔄 Reset Conversation", use_container_width=True):
            reset_simulation()
            st.rerun()

def display_conversation_history(child_name: str):
    """Display the conversation history"""
    st.markdown("<h3 class='subsection-header'>Conversation:</h3>", unsafe_allow_html=True)
    for msg in st.session_state['conversation_history']:
        col1, col2 = st.columns([8, 4])
        with col1:
            speaker = "You" if msg['role'] == 'parent' else child_name
            st.markdown(f"""
                <div class='message-{msg["role"]}'>
                    <strong>{speaker}:</strong> {msg['content']}
                </div>
            """, unsafe_allow_html=True)
        with col2:
            if msg['role'] == 'parent' and 'feedback' in msg:
                st.info(f"💡 {msg['feedback']}")

def display_saved_reflections(prolific_id: str):
    """Display user's saved reflections"""
    st.markdown("<h2 class='section-header'>Your Saved Reflections</h2>", unsafe_allow_html=True)
    
    if not prolific_id:
        st.warning("Please ensure your Prolific ID is entered correctly to view reflections.")
        return
    
    reflections = supabase_manager.get_reflections(prolific_id)
    
    if not reflections:
        st.info("No reflections found. Complete a role-play simulation and save your reflections to see them here.")
        return
    
    track_feature_visit("View Reflections")
    
    st.write(f"Found {len(reflections)} saved reflection(s)")
    
    for reflection in reflections:
        with st.expander(f"Reflection from {datetime.fromisoformat(reflection['created_at']).strftime('%Y-%m-%d %H:%M')}"):
            try:
                content = reflection['content']
                display_reflection_content(content)
            except Exception as e:
                print(f"Error displaying reflection: {e}")
                st.error("Error displaying reflection content.")

def handle_conversation_input(send_button: bool, end_button: bool, user_input: str, child_age: str, situation: str):
    if send_button and user_input:
        feedback = provide_realtime_feedback(user_input, st.session_state['strategy'])
        
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
        supabase_manager.save_simulation_data(simulation_data)
        
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
        
        # Random mood change
        if random.random() < 0.3:
            st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
            update_langsmith_run(
                st.session_state['run_id'],
                {"mood_change": st.session_state['child_mood']}
            )
        
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

def handle_user_input(child_age: str, situation: str):
    """Handle user input during conversation"""
    # Move form creation to this level
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
    
    if st.session_state.get('paused'):
        st.info("""
            Take a moment to reflect:
            - What emotions are you noticing in yourself?
            - What might your child be feeling?
            - What's your goal in this interaction?
        """)
        return
        
    if st.session_state.get('show_hints'):
        st.info("\n".join(STRATEGY_HINTS[st.session_state['strategy']]))
    
    handle_conversation_input(send_button, end_button, user_input, child_age, situation)

def display_conversation_playback(conversation_history):
    """Display the conversation history with feedback"""
    st.markdown("<h2 class='section-header'>Conversation Playback</h2>", unsafe_allow_html=True)
    
    if not conversation_history:
        st.info("No conversation history to display.")
        return
        
    for msg in conversation_history:
        with st.expander(f"{msg['role'].title()}'s message"):
            st.write(msg['content'])
            if msg.get('feedback'):
                st.info(f"💡 Feedback: {msg['feedback']}")
            if msg.get('strategy_used'):
                st.info(f"Strategy used: {msg['strategy_used']}")

def end_simulation(conversation_history: list, child_age: str, strategy: str):
    """Handle end of simulation and gather reflections"""
    st.session_state['simulation_ended'] = True
    
    prolific_id = st.session_state.get('parent_name')
    if not prolific_id:
        st.error("Error: Prolific ID not found. Please ensure you've entered your ID correctly.")
        return
        
    track_feature_visit("Role-Play Simulation")
    
    st.write("The simulation has ended.")

    # Get used strategies
    strategies_used = {strategy}
    if conversation_history:
        strategies_used.update(msg.get('strategy_used', strategy) 
                             for msg in conversation_history 
                             if msg.get('role') == 'parent' and msg.get('strategy_used'))

    display_conversation_playback(conversation_history)
    gather_end_simulation_reflections(strategies_used, prolific_id)

def gather_end_simulation_reflections(strategies_used: set, prolific_id: str):
    """Gather reflections at the end of simulation"""
    st.markdown("<h2 class='section-header'>Final Reflection</h2>", unsafe_allow_html=True)
    
    with st.form(key='end_simulation_form'):
        current_reflection = {}
        
        for strategy_used in strategies_used:
            current_reflection[f"How effective was {strategy_used} in this conversation?"] = st.text_area(
                label=f"How effective was {strategy_used} in this conversation?",
                height=100,
                key=f"reflection_strategy_{strategy_used}"
            )
        
        current_reflection["What did you learn about your child's perspective?"] = st.text_area(
            label="What did you learn about your child's perspective?",
            height=100,
            key="reflection_perspective"
        )
        
        current_reflection["What would you do differently next time?"] = st.text_area(
            label="What would you do differently next time?",
            height=100,
            key="reflection_improvements"
        )
        
        submit_button = st.form_submit_button("Save Reflection")
        
        if submit_button:
            handle_reflection_submission(current_reflection, strategies_used, prolific_id)

def display_reflection_content(content: dict):
    """Display reflection content in a structured format"""
    if 'strategies_used' in content:
        st.write("**Strategies used:**")
        st.write(", ".join(content['strategies_used']))
        st.write("")

    if 'reflection_content' in content:
        st.write("**Your Reflections:**")
        for question, answer in content['reflection_content'].items():
            if answer and answer.strip():
                st.markdown(f"""
                    <div class='reflection-item'>
                        <strong>{question}</strong><br>
                        {answer}
                    </div>
                    <hr>
                """, unsafe_allow_html=True)

    if 'conversation_summary' in content:
        summary = content['conversation_summary']
        st.write(f"\n**Conversation Length:** {summary.get('length', 'N/A')} turns")

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
        - 💭 **Conversation Starters** - Learn how to begin difficult conversations
        - 🗣️ **Communication Techniques** - Discover effective communication strategies
        - 🎮 **Role-Play Simulation** - Practice conversations in a safe environment
        - 📝 **Learning Reflections** - Track your progress and insights
    """)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Got it, let's start!", use_container_width=True):
            st.session_state.show_tutorial = False
            st.rerun()

def handle_reflection_submission(current_reflection, strategies_used, prolific_id):
    print("Current reflection:", current_reflection)
    print("Strategies used:", strategies_used)
    print("Prolific ID:", prolific_id)
    reflection_data = {
        "user_id": prolific_id,
        "type": "end_simulation",
        "content": {
            "reflection_content": current_reflection,
            "strategies_used": list(strategies_used),
            "conversation_summary": {
                "length": st.session_state['turn_count']
            }
        }
    }
    
    if supabase_manager.save_reflection(prolific_id, "end_simulation", reflection_data):
        st.success("Reflection saved successfully!")
    else:
        st.error("Failed to save reflection")

def reset_simulation():
    st.session_state['conversation_history'] = []
    st.session_state['turn_count'] = 0
    st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
    st.session_state['simulation_id'] = str(uuid4())


def main():
    """Main application entry point"""
    check_environment()
    
    # Initialize feature order and descriptions
    feature_order = {
        "Advice": "Get expert guidance on handling specific parenting situations based on evidence-based strategies.",
        "Conversation Starters": "Receive help initiating difficult conversations with suggested opening phrases and questions.",
        "Communication Techniques": "Discover helpful ways to talk with your child and get tips on how to give clear and simple answers.",
        "Role-Play Simulation": "Practice conversations in a safe environment to develop and refine your communication approach.",
        "View Reflections": "Review your saved insights and learning from previous practice sessions."
    }

    if not st.session_state.get('info_submitted', False):
        show_info_screen()
    else:
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
            elif selected == "Conversation Starters":
                track_feature_visit("conversation_starters")
                display_conversation_starters(st.session_state['situation'])
            elif selected == "Communication Techniques":
                track_feature_visit("communication_techniques")
                display_communication_techniques(st.session_state['situation'])
            elif selected == "Role-Play Simulation":
                track_feature_visit("role_play")
                simulate_conversation_streamlit(st.session_state['parent_name'], st.session_state['child_age'], st.session_state['situation'])
            elif selected == "View Reflections":
                track_feature_visit("reflections")
                display_saved_reflections(st.session_state['parent_name'])
                display_stored_data()  # Add this line

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