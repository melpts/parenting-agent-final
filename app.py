import streamlit as st
import os
import openai
from dotenv import load_dotenv
import json
import re
import sqlite3
from datetime import datetime
import random
from uuid import UUID, uuid4
import warnings
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from langsmith import Client
from langsmith.run_helpers import traceable

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

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

from streamlit_feedback import streamlit_feedback
from langsmith import Client
from langsmith.run_helpers import traceable
from langsmith.run_helpers import get_current_run_tree

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Parenting Support Bot",
    initial_sidebar_state="expanded"
)

# Environment Variables Setup
def setup_environment():
    load_dotenv()
    
    if hasattr(st, 'secrets'):
        openai.api_key = st.secrets.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = st.secrets.get('OPENAI_API_KEY', '')
        os.environ['LANGCHAIN_API_KEY'] = st.secrets.get('LANGCHAIN_API_KEY', '')
        os.environ['LANGCHAIN_PROJECT'] = st.secrets.get('LANGCHAIN_PROJECT', 'Parenting agent2')
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    else:
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            st.error('OpenAI API key not found! Please set it in your environment or Streamlit secrets.')
            st.stop()

# Call setup at start
setup_environment()

def check_api_keys():
    """Verify that all required API keys are present"""
    missing_keys = []
    
    if not openai.api_key:
        missing_keys.append("OpenAI API Key")
    
    if not os.getenv('LANGCHAIN_API_KEY'):
        missing_keys.append("LangChain API Key")
        
    if missing_keys:
        st.error(f"Missing required API keys: {', '.join(missing_keys)}")
        st.info("Please add the missing API keys to your Streamlit secrets or environment variables.")
        st.stop()

# Initialize LangSmith Client
smith_client = Client()

# Database setup with updated UUID handling
DATABASE_URL = "sqlite:///parenting_app.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Reflection(Base):
    __tablename__ = "reflections"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    type = Column(String)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    langsmith_run_id = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

def create_langsmith_run(name, inputs, fallback_id=None):
    try:
        run = smith_client.create_run(
            run_type="chain",
            name=name,
            inputs=inputs
        )
        return str(run.id) if run else str(uuid4())
    except Exception as e:
        print(f"Error creating LangSmith run: {e}")
        return str(uuid4())

def update_langsmith_run(run_id, outputs):
    if not run_id:
        return
    try:
        try:
            UUID(run_id)  # Validate UUID format
            smith_client.update_run(run_id, outputs=outputs)
        except ValueError:
            new_uuid = str(uuid4())
            print(f"Invalid UUID {run_id}, using new UUID: {new_uuid}")
            smith_client.update_run(new_uuid, outputs=outputs)
    except Exception as e:
        print(f"Error updating LangSmith run: {e}")

def setup_memory():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(
        chat_memory=msgs,
        return_messages=True,
        memory_key="history"
    )
    return memory

# Initialize memory
memory = setup_memory()

from conversation_starter_citations import CONVERSATION_STARTER_CITATIONS
from communication_strategies_citations import COMMUNICATION_STRATEGIES_CITATIONS
from simulation_citations import SIMULATION_CITATIONS
from Website_citations import WEBSITE_CITATIONS
from Active_listening_citations import ACTIVE_LISTENING_CITATIONS
from i_messages_citations import I_MESSAGES_CITATIONS
from positive_reinforcement import POSITIVE_REINFORCEMENT_CITATIONS
from Reflective_questioning import REFLECTIVE_QUESTIONING_CITATIONS

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
            "But I'm not done yet!",
            "*watching TV* After this part..."
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

def init_session_state():
    if 'run_id' not in st.session_state:
        st.session_state['run_id'] = str(uuid4())
    if 'agentState' not in st.session_state:
        st.session_state['agentState'] = "start"
    if 'consent' not in st.session_state:
        st.session_state['consent'] = False
    if 'exp_data' not in st.session_state:
        st.session_state['exp_data'] = True
    if 'llm_model' not in st.session_state:
        st.session_state['llm_model'] = "gpt-4"
    if 'simulation_ended' not in st.session_state:
        st.session_state['simulation_ended'] = False
    if 'stored_responses' not in st.session_state:
        st.session_state['stored_responses'] = {}
    if 'show_hints' not in st.session_state:
        st.session_state['show_hints'] = False
    if 'paused' not in st.session_state:
        st.session_state['paused'] = False
    if 'info_submitted' not in st.session_state:
        st.session_state['info_submitted'] = False
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    if 'child_mood' not in st.session_state:
        st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
    if 'turn_count' not in st.session_state:
        st.session_state['turn_count'] = 0
    if 'strategy' not in st.session_state:
        st.session_state['strategy'] = "Active Listening"
    if 'simulation_id' not in st.session_state:
        st.session_state['simulation_id'] = str(uuid4())

def track_feature_visit(feature_name):
    if 'visited_features' not in st.session_state:
        st.session_state.visited_features = set()
    
    # Normalize the feature name to match the display format
    feature_display_map = {
        "advice": "Advice",
        "conversation_starters": "Conversation Starters",
        "communication_techniques": "Communication Techniques",
        "role_play": "Role-Play Simulation",
        "reflections": "View Reflections",
        "Role-Play Simulation": "Role-Play Simulation",
        "View Reflections": "View Reflections"
    }
    
    # Add both the normalized and display versions
    normalized_name = feature_display_map.get(feature_name, feature_name)
    st.session_state.visited_features.add(normalized_name)
    
    # Also add the lowercase version for compatibility
    st.session_state.visited_features.add(normalized_name.lower().replace(" ", "_"))

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

    /* Form Styling */
    .stTextArea > div > div > textarea {
        font-size: 1.1em;
        line-height: 1.6;
        padding: 1em;
        border: 2px solid #f3f4f6;
        border-radius: 0.5em;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
    }
    </style>
"""

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
init_session_state()

@traceable(name="generate_child_response")
def generate_child_response(conversation_history, child_age, situation, mood, strategy, parent_response):
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

def provide_realtime_feedback(parent_response, strategy):
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

def reformulate_phrase(phrase, strategy):
    messages = [
        {"role": "system", "content": f"You are a parenting communication expert. Reformulate the given phrase using {strategy} principles. Keep the same meaning but change the delivery."},
        {"role": "user", "content": f"Please reformulate this phrase: {phrase}"}
    ]
    
    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error reformulating phrase: {e}")
        return None

def generate_conversation_starters(situation):
    prompt = f"""
    SYSTEM
    Use the provided citations delimited by triple quotes to answer questions. If the answer cannot be found in the citations, write "I could not find an answer."
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

def save_reflection(user_id, reflection_type, content):
    if not user_id:
        st.warning("Please ensure your Prolific ID is entered correctly.")
        return False
        
    try:
        db = SessionLocal()
        
        # Ensure content is properly serialized
        if isinstance(content, dict):
            content_str = json.dumps(content)
        else:
            content_str = json.dumps({"content": str(content)})
        
        # Create new reflection
        db_reflection = Reflection(
            user_id=user_id,
            type=reflection_type,
            content=content_str,
            langsmith_run_id=st.session_state.get('run_id'),
            timestamp=datetime.utcnow()
        )
        
        # Save to database
        db.add(db_reflection)
        db.commit()
        db.refresh(db_reflection)
        
        print(f"Successfully saved reflection with content: {content_str}")
        return True
        
    except Exception as e:
        print(f"Error saving reflection for Prolific ID {user_id}: {str(e)}")
        st.error(f"Failed to save reflection: {str(e)}")
        return False
    finally:
        db.close()

def reset_simulation():
    st.session_state['conversation_history'] = []
    st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
    st.session_state['turn_count'] = 0
    st.session_state['strategy'] = "Active Listening"
    st.session_state['simulation_ended'] = False
    st.session_state['simulation_id'] = str(uuid4())
    st.session_state['stored_responses'].clear()
    st.session_state['run_id'] = str(uuid4())

def show_info_screen():
    st.markdown("<h1 class='main-header'>Welcome to Parenting Support Bot</h1>", unsafe_allow_html=True)
    
    with st.form(key='parent_info_form'):
        st.markdown("<h2 class='section-header'>Please Tell Us About You</h2>", unsafe_allow_html=True)
        
        st.markdown("""
            <p class='description-text'>
                Please enter your <b>Prolific ID</b> (24-character identifier from your Prolific account)
            </p>
        """, unsafe_allow_html=True)
        
        parent_name = st.text_input(
            label="Prolific ID",
            placeholder="Enter your 24-character Prolific ID...",
            help="This is the ID assigned to you by Prolific, found in your Prolific account"
        )
        
        child_name = st.text_input(
            label="Child's Name",
            placeholder="Enter your child's name..."
        )
        
        age_ranges = ["3-5 years", "6-9 years", "10-12 years"]
        child_age = st.selectbox(
            label="Child's Age Range",
            options=age_ranges
        )
        
        situation = st.text_area(
            label="Situation Description",
            placeholder="Type your situation here...",
            height=120,
            label_visibility="visible"
        )
        
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

@traceable(name="simulate_conversation")
def simulate_conversation_streamlit(name, child_age, situation):
    name = st.session_state.get('parent_name', name)
    child_name = st.session_state.get('child_name', '')
    child_age = st.session_state.get('child_age', child_age)
    situation = st.session_state.get('situation', situation)

    # Title
    st.markdown("## Parent-Child Role-Play Simulator")

    # Instructions in a clean container
    with st.container():
        st.markdown("""
            #### How to use this simulator:
            1. Start by responding naturally to your child's situation
            2. Try different communication strategies to see their impact
            3. Use the conversation controls to get hints or pause for reflection
        """)

    # Current situation in an info box
    st.info(f"**Current Situation:** {situation}")

    # Initialize conversation state
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
        st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
        st.session_state['turn_count'] = 0
        st.session_state['strategy'] = "Active Listening"
        st.session_state['simulation_id'] = str(uuid4())
        
        run_id = create_langsmith_run(
            name="parenting_conversation",
            inputs={
                "parent_name": name,
                "child_name": child_name,
                "child_age": child_age,
                "situation": situation,
                "initial_strategy": "Active Listening"
            }
        )
        st.session_state['run_id'] = run_id

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
    
    # Display current strategy explanation
    st.markdown(STRATEGY_EXPLANATIONS[st.session_state['strategy']], unsafe_allow_html=True)

    # Action buttons
    st.markdown("### Conversation Controls")
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("⏸️ Pause/Resume", use_container_width=True):
            st.session_state['paused'] = not st.session_state['paused']
    with action_cols[1]:
        if st.button("💡 Show Hints", use_container_width=True):
            st.session_state['show_hints'] = not st.session_state['show_hints']
    with action_cols[2]:
        if st.button("🔄 Reformulate Response", use_container_width=True):
            if st.session_state['conversation_history']:
                last_response = next((msg['content'] for msg in reversed(st.session_state['conversation_history']) 
                                   if msg['role'] == 'parent'), None)
                if last_response:
                    reformulated = reformulate_phrase(last_response, st.session_state['strategy'])
                    if reformulated:
                        st.info(f"Reformulated response suggestion:\n\n{reformulated}")

    # Handle pause state
    if st.session_state['paused']:
        st.info("""
        Take a moment to reflect:
        - What emotions are you noticing in yourself?
        - What might your child be feeling?
        - What's your goal in this interaction?
        """)
        if st.button("Continue"):
            st.session_state['paused'] = False
            st.rerun()
        return

    # Show hints if requested
    if st.session_state['show_hints']:
        hints = STRATEGY_HINTS[st.session_state['strategy']]
        st.info("\n".join(hints))

    # Display conversation
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

    # Parent's input section
    with st.form(key=f'parent_input_form_{st.session_state["simulation_id"]}_{st.session_state["turn_count"]}'):
        user_input = st.text_area(
            label="Your response",
            placeholder="How would you start this conversation with your child? Type here...",
            key=f"parent_input_{st.session_state['simulation_id']}_{st.session_state['turn_count']}",
            height=100,
            label_visibility="visible"
        )
        submit_cols = st.columns(2)
        with submit_cols[0]:
            send_button = st.form_submit_button("Send Response", use_container_width=True)
        with submit_cols[1]:
            end_button = st.form_submit_button("End Conversation", use_container_width=True, type="secondary")

    # Handle user input
    handle_conversation_input(send_button, end_button, user_input, child_age, situation)

def handle_conversation_input(send_button, end_button, user_input, child_age, situation):
    if send_button and user_input:
        feedback = provide_realtime_feedback(user_input, st.session_state['strategy'])
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
        
        child_response = generate_child_response(
            st.session_state['conversation_history'],
            child_age,
            situation,
            st.session_state['child_mood'],
            st.session_state['strategy'],
            user_input
        )
        
        st.session_state['conversation_history'].append({
            "role": "child",
            "content": child_response,
            "id": len(st.session_state['conversation_history'])
        })
        
        if random.random() < 0.3:  # 30% chance to change mood
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

def end_simulation(conversation_history, child_age, strategy):
    st.session_state['simulation_ended'] = True
    
    if 'parent_name' not in st.session_state:
        st.error("Error: Prolific ID not found. Please ensure you've entered your ID correctly.")
        return
        
    prolific_id = st.session_state['parent_name']
    
    # Track that role-play simulation was completed
    track_feature_visit("Role-Play Simulation")
    
    st.write("The simulation has ended.")

    # Initialize strategies_used set
    strategies_used = {strategy}  # Start with the current strategy
    if conversation_history:
        # Add any strategies used during the conversation
        strategies_used.update(msg.get('strategy_used', strategy) 
                             for msg in conversation_history 
                             if msg.get('role') == 'parent' and msg.get('strategy_used'))

    st.markdown("<h2 class='section-header'>Conversation Playback</h2>", unsafe_allow_html=True)
    if conversation_history:
        for msg in conversation_history:
            with st.expander(f"{msg['role'].title()}'s message"):
                st.write(msg['content'])
                if msg.get('feedback'):
                    st.info(f"💡 Feedback: {msg['feedback']}")
            
    st.markdown("<h2 class='section-header'>Final Reflection</h2>", unsafe_allow_html=True)
    
    with st.form(key='end_simulation_form'):
        current_reflection = {}
        
        for strategy_used in strategies_used:
            current_reflection[f"How effective was {strategy_used} in this conversation?"] = st.text_area(
                label=f"How effective was {strategy_used} in this conversation?",
                height=100,
                key=f"reflection_strategy_{strategy_used}",
                label_visibility="visible"
            )
        
        current_reflection["What did you learn about your child's perspective?"] = st.text_area(
            label="What did you learn about your child's perspective?",
            height=100,
            key="reflection_perspective",
            label_visibility="visible"
        )
        
        current_reflection["What would you do differently next time?"] = st.text_area(
            label="What would you do differently next time?",
            height=100,
            key="reflection_improvements",
            label_visibility="visible"
        )
        
        submit_button = st.form_submit_button("Save Reflection")
    
    if submit_button:
        if not any(answer.strip() for answer in current_reflection.values()):
            st.warning("Please fill in at least one reflection question before saving.")
            return
            
        reflection_data = {
            'reflection_content': current_reflection,
            'strategies_used': list(strategies_used),
            'conversation_summary': {
                'length': len(conversation_history),
                'strategies_used': list(strategies_used),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        success = save_reflection(prolific_id, 'end_simulation', reflection_data)
        
        if success:
            # Track that reflection was saved
            track_feature_visit("View Reflections")
            st.success("✨ Reflection saved successfully! View it in the 'View Reflections' tab.")
            if st.button("Start New Conversation"):
                reset_simulation()
                st.rerun()
        else:
            st.error("Failed to save reflection. Please try again.")

@traceable(name="display_advice")
def display_advice(parent_name, child_age, situation):
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

@traceable(name="display_conversation_starters")
def display_conversation_starters(situation):
    st.markdown("<h2 class='section-header'>Conversation Starters</h2>", unsafe_allow_html=True)
    if situation:
        try:
            with st.spinner('Generating conversation starters...'):
                run_id = create_langsmith_run(
                    name="conversation_starters",
                    inputs={"situation": situation},
                    fallback_id=str(random.randint(1000, 9999))
                )
                
                starters = generate_conversation_starters(situation)
                
                update_langsmith_run(run_id, {"starters": starters})
                st.write(starters)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please describe the situation in the sidebar to get conversation starters.")

@traceable(name="display_communication_techniques")
def display_communication_techniques(situation):
    st.markdown("<h2 class='section-header'>Communication Techniques</h2>", unsafe_allow_html=True)
    if situation:
        try:
            with st.spinner('Generating communication techniques...'):
                run_id = create_langsmith_run(
                    name="communication_techniques",
                    inputs={"situation": situation},
                    fallback_id=str(random.randint(1000, 9999))
                )
                
                messages = [
                    {"role": "system", "content": "You are a parenting expert focused on effective communication strategies."},
                    {"role": "user", "content": f"Provide specific communication techniques for this situation: {situation}"}
                ]
                completion = openai.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )
                
                response = completion.choices[0].message.content
                update_langsmith_run(run_id, {"techniques": response})
                
                st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please describe the situation in the sidebar to get communication techniques.")

def display_saved_reflections(prolific_id):
    st.markdown("<h2 class='section-header'>Your Saved Reflections</h2>", unsafe_allow_html=True)
    
    if not prolific_id:
        st.warning("Please ensure your Prolific ID is entered correctly to view reflections.")
        return
        
    print(f"Attempting to load reflections for Prolific ID: {prolific_id}")
    
    db = SessionLocal()
    try:
        # Add ORDER BY to show newest first
        reflections = db.query(Reflection).filter(
            Reflection.user_id == prolific_id,
            Reflection.type == 'end_simulation'  # Only show end simulation reflections
        ).order_by(Reflection.timestamp.desc()).all()
        
        if not reflections:
            st.info("No reflections found. Complete a role-play simulation and save your reflections to see them here.")
            return
        
        # Mark View Reflections as visited if there are reflections to display
        track_feature_visit("View Reflections")
        
        st.write(f"Found {len(reflections)} saved reflection(s)")
        
        for reflection in reflections:
            with st.expander(f"Reflection from {reflection.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                try:
                    content = json.loads(reflection.content)
                    if isinstance(content, dict):
                        # Display strategies used
                        if 'strategies_used' in content:
                            st.write("**Strategies used:**")
                            st.write(", ".join(content['strategies_used']))
                            st.write("")

                        # Display reflection content
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

                        # Display conversation summary if available
                        if 'conversation_summary' in content:
                            summary = content['conversation_summary']
                            st.write(f"\n**Conversation Length:** {summary.get('length', 'N/A')} turns")
                            
                except json.JSONDecodeError as e:
                    print(f"Error parsing reflection content: {str(e)}")
                    st.error("Error loading reflection content.")
                    print(f"Raw content: {reflection.content}")
                except Exception as e:
                    print(f"Error displaying reflection: {str(e)}")
                    st.error("Error displaying reflection.")
                    print(f"Raw content: {reflection.content}")
    finally:
        db.close()

def main():
    # Initialize feature order and descriptions at the start of the function
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
                if 'conversation_history' in st.session_state:
                    st.session_state.pop('conversation_history')
                if 'run_id' in st.session_state:
                    st.session_state.pop('run_id')
                st.rerun()

        # Show tutorial if it's the first time
        if 'show_tutorial' not in st.session_state:
            st.session_state.show_tutorial = True
            st.session_state.show_features = False
        
        # Display either tutorial or features
        if st.session_state.show_tutorial:
            st.markdown("# Welcome to the Parenting Support Bot! 🎉", unsafe_allow_html=True)
            
            st.markdown("""
                This app is designed to help you develop effective parenting strategies through:
            """)
            
            st.markdown("""
                - 📚 **Expert Advice** - Get evidence-based parenting advice
                - 💭 **Conversation Starters** - Learn how to begin difficult conversations
                - 🗣️ **Communication Techniques** - Discover effective communication strategies
                - 🎮 **Role-Play Simulation** - Practice conversations in a safe environment before speaking with your child
                - 📝 **Learning Reflections** - Track your progress and insights through reflections 
            """)
            
            st.markdown("""
                We recommend exploring each feature in order, but feel free to use them however works best for you!
            """)
            
            # Center the button using columns
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("Got it, let's start!", use_container_width=True):
                    st.session_state.show_tutorial = False
                    st.rerun()
        else:
            # Show main title after tutorial
            st.markdown("<h1 class='main-header'>Parenting Support Bot</h1>", unsafe_allow_html=True)

            # Feature selection
            selected = st.radio(
                "Choose an option:",
                list(feature_order.keys()),
                horizontal=True,
                help="Select a tool that best matches your current needs",
                label_visibility="visible"
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

            # Add progress indicator in sidebar
            if 'visited_features' in st.session_state:
                st.sidebar.markdown("### Your Progress")
                for feature in feature_order.keys():
                    feature_key = feature.lower().replace(" ", "_")
                    if feature_key in st.session_state.visited_features:
                        st.sidebar.markdown(f"✅ {feature}")
                    else:
                        st.sidebar.markdown(f"◽ {feature}")

def setup_database():
    conn = sqlite3.connect('user_queries.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  user_query TEXT,
                  bot_response TEXT)''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()
    main()