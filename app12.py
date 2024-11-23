import streamlit as st

st.set_page_config(
    layout="wide", 
    page_title="Parenting Support Bot",
    initial_sidebar_state="expanded"
)

#custom CSS 
st.markdown("""
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
""", unsafe_allow_html=True)

import os
import json
import re
import openai
import sqlite3
from datetime import datetime
import random
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from langsmith import Client
from langsmith.run_helpers import traceable

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

# Load environment variables
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

# Import citation variables
from conversation_starter_citations import CONVERSATION_STARTER_CITATIONS
from communication_strategies_citations import COMMUNICATION_STRATEGIES_CITATIONS
from simulation_citations import SIMULATION_CITATIONS
from Website_citations import WEBSITE_CITATIONS
from Active_listening_citations import ACTIVE_LISTENING_CITATIONS
from i_messages_citations import I_MESSAGES_CITATIONS
from positive_reinforcement import POSITIVE_REINFORCEMENT_CITATIONS
from Reflective_questioning import REFLECTIVE_QUESTIONING_CITATIONS

# Set up environment variables and API keys
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
openai.api_key = os.environ["OPENAI_API_KEY"]

os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"] = st.secrets['LANGCHAIN_PROJECT']
os.environ["LANGCHAIN_PROJECT"] = "Parenting agent2"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize LangSmith Client
smith_client = Client()

# Database setup
DATABASE_URL = "sqlite:///parenting_app.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Reflection(Base):
    __tablename__ = "reflections"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    type = Column(String)
    content = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    langsmith_run_id = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

# LangSmith Helper Functions
def create_langsmith_run(name, inputs, fallback_id=None):
    try:
        run = smith_client.create_run(
            run_type="chain",
            name=name,
            inputs=inputs
        )
        return run.id if run else fallback_id
    except Exception as e:
        print(f"Error creating LangSmith run: {e}")
        return fallback_id

def update_langsmith_run(run_id, outputs):
    if not run_id:
        return
    try:
        smith_client.update_run(run_id, outputs=outputs)
    except Exception as e:
        print(f"Error updating LangSmith run: {e}")

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

# Age-specific responses dictionary
AGE_SPECIFIC_RESPONSES = {
    "3-5 years": {
        "cooperative": [
            "Okay, I'll try...",
            "Can you help me?",
            "I want to be good!",
            "Like this, Mommy/Daddy?",
            "It's hard!"
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
            "I want MY way!"
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
            "Is it time for cartoons?"
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
            "Will you show me how?"
            "*putting down game* Okay, I understand",
            "I know I should... *starting task*",
            "*organizing things* I'm helping!",
            "You're right... *beginning task*",
            "*showing work* Is this better?",
        ],
        "defiant": [
            "But that's not fair! Jamie never has to!",
            "*arms crossed* You can't make me!",
            "I hate these rules!",
            "You never let me do anything fun!",
            "Well, Sarah's parents let her!"
            "But Emma's mom lets her!",
            "*rolling eyes* This isn't fair!",
            "You NEVER let me do anything fun!",
            "*slamming door* Leave me alone!",
            "Why do I always have to?",
            "*arms crossed* Make me!",
            "You're the worst! *storming off*",
        ],
        "distracted": [
            "But first can I just...",
            "Wait, I forgot to tell you about...",
            "Can we do it later? I'm almost finished with...",
            "Oh! I just remembered something!"
            "*playing game* Just need to save...",
            "But first can I...",
            "Did you know that... *changing subject*",
            "*watching YouTube* Almost done!",
            "Wait, I forgot to tell you about...",
            "*fixated on phone* In a minute...",
        ]
    },
    "10-12 years": {
        "cooperative": [
            "Fine, I get it. Just give me a minute.",
            "I know, I know. I'm going.",
            "Okay, but can we talk about it first?",
            "I understand, but...",
            "I'll do it, just let me finish this."
            "*putting phone down* Okay, I'm listening",
            "Can we talk about it?",
            "*nodding* That's fair",
            "I understand... *complying*",
            "*showing compromise* How about this?",
        ],
        "defiant": [
            "This is so unfair! You never understand!",
            "Everyone else gets to!",
            "*slamming door* Leave me alone!",
            "You're ruining everything!"
            "*eye roll* Whatever",
            "This is SO unfair! *texting friends*",
            "You don't understand ANYTHING!",
            "*slamming door* I hate this!",
            "Everyone else's parents...",
            "*storming off* You're ruining my life!",
            "This is stupid! *throwing phone*",
        ],
        "distracted": [
            "Yeah, just one more level...",
            "Hold on, I'm texting...",
            "In a minute... I'm doing something.",
            "But I'm in the middle of something!"
            "*texting* Just one sec...",
            "But my friends are waiting online...",
            "*watching TikTok* Almost done",
            "*gaming* Can't pause multiplayer!",
            "Hold on, I'm in the middle of...",
            "*scrolling phone* Five more minutes?",
        ]
    }
}

# Initialize session state
if 'run_id' not in st.session_state: 
    st.session_state['run_id'] = None
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

# Setup chat memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

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

setup_database()

def save_reflection(user_id, reflection_type, content):
    if not user_id or user_id == 'Anonymous':
        st.warning("Please enter your name in the sidebar to save reflections.")
        return False
        
    try:
        db = SessionLocal()
        
        # Ensure content is JSON serializable
        if isinstance(content, dict):
            content_str = json.dumps(content)
        else:
            content_str = json.dumps({"content": content})
        
        db_reflection = Reflection(
            user_id=user_id,
            type=reflection_type,
            content=content_str,
            langsmith_run_id=st.session_state.get('run_id'),
            timestamp=datetime.utcnow()
        )
        
        db.add(db_reflection)
        db.commit()
        db.refresh(db_reflection)
        
        if st.session_state.get('run_id'):
            update_langsmith_run(
                st.session_state['run_id'],
                {
                    "reflection_saved": {
                        "type": reflection_type,
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            )
        
        return True
        
    except Exception as e:
        print(f"Error saving reflection: {str(e)}")
        st.error(f"Failed to save reflection: {str(e)}")
        return False
    finally:
        db.close()

def load_reflections(user_id):
    db = SessionLocal()
    try:
        reflections = db.query(Reflection).filter(Reflection.user_id == user_id).order_by(Reflection.timestamp.desc()).all()
        return reflections
    except Exception as e:
        print(f"Error loading reflections: {str(e)}")
        return []
    finally:
        db.close()

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
        
        Analyze parent's response:
        - If stern/commanding: Show appropriate resistance or compliance based on mood
        - If supportive/understanding: React with more openness
        - If asking questions: Give age-appropriate answers that match emotional state
        - Stay consistent with your previous responses in the conversation

        Parent's response to react to: {parent_response}"""}
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
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()

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

def show_info_screen():
    st.markdown("<h1 class='main-header'>Welcome to Parenting Support Bot</h1>", unsafe_allow_html=True)
    
    with st.form(key='parent_info_form'):
        st.markdown("<h2 class='section-header'>Please Tell Us About You</h2>", unsafe_allow_html=True)
        parent_name = st.text_input(
            "Prolific ID",
            placeholder="Enter your Prolific ID...",
        )
        child_name = st.text_input(
            "Child's Name",
            placeholder="Enter your child's name...",
        )
        
        age_ranges = ["3-5 years", "6-9 years", "10-12 years"]
        child_age = st.selectbox("Child's Age Range", age_ranges)
        
        st.markdown("<p class='description-text'>Describe the situation you'd like help with:</p>", 
                   unsafe_allow_html=True)
        situation = st.text_area(
            "",
            placeholder="Type your situation here...",
            height=120
        )
        
        submit_button = st.form_submit_button("Start", use_container_width=True)
        
        if submit_button and parent_name and child_name and situation:
            st.session_state['parent_name'] = parent_name
            st.session_state['child_name'] = child_name
            st.session_state['child_age'] = child_age
            st.session_state['situation'] = situation
            st.session_state['info_submitted'] = True
            st.rerun()
        elif submit_button:
            st.error("Please fill in all fields")
            
@traceable(name="simulate_conversation")
def simulate_conversation_streamlit(name, child_age, situation):
    name = st.session_state.get('parent_name', name)
    child_name = st.session_state.get('child_name', '')
    child_age = st.session_state.get('child_age', child_age)
    situation = st.session_state.get('situation', situation)
    
    st.markdown("<h1 class='main-header'>Parent-Child Role-Play Simulator</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class='description-text'>
            Welcome to the conversation simulator! Here you can practice different communication strategies 
            with your child in a safe environment. Start by responding to the situation as you normally would, 
            and then try incorporating different communication strategies to see how they might change the interaction.
        </div>
        
        <div class='situation-text'>
            <strong>Your situation:</strong> {situation}
        </div>
    """, unsafe_allow_html=True)

    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
        st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
        st.session_state['turn_count'] = 0
        st.session_state['strategy'] = "Active Listening"
        st.session_state['simulation_id'] = random.randint(1000, 9999)

        run_id = create_langsmith_run(
            name="parenting_conversation",
            inputs={
                "parent_name": name,
                "child_name": child_name,
                "child_age": child_age,
                "situation": situation,
                "initial_strategy": "Active Listening"
            },
            fallback_id=str(st.session_state['simulation_id'])
        )
        st.session_state['run_id'] = run_id

    # Strategy selection with improved layout
    st.markdown("<h2 class='section-header'>Choose your communication strategy:</h2>", unsafe_allow_html=True)
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

    # Action buttons with improved styling
    st.markdown("<h3 class='subsection-header'>Conversation Controls</h3>", unsafe_allow_html=True)
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

    # If paused, show reflection prompts
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

    # Display conversation with improved formatting
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

    # Parent's input section with placeholder text
    with st.form(key=f'parent_input_form_{st.session_state["simulation_id"]}_{st.session_state["turn_count"]}'):
        user_input = st.text_area(
            "Your response:",
            placeholder="Type your response here...",
            key=f"parent_input_{st.session_state['simulation_id']}_{st.session_state['turn_count']}",
            height=100
        )
        submit_cols = st.columns(2)
        with submit_cols[0]:
            send_button = st.form_submit_button("Send Response", use_container_width=True)
        with submit_cols[1]:
            end_button = st.form_submit_button("End Conversation", use_container_width=True, type="secondary")

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
    st.write("The simulation has ended.")

    # Add interaction playback
    st.markdown("<h2 class='section-header'>Conversation Playback</h2>", unsafe_allow_html=True)
    if conversation_history:
        for msg in conversation_history:
            with st.expander(f"{msg['role'].title()}'s message"):
                st.write(msg['content'])
                if msg.get('feedback'):
                    st.info(f"💡 Feedback: {msg['feedback']}")

        # Add communication pattern analysis
        st.markdown("<h2 class='section-header'>Communication Pattern Analysis</h2>", unsafe_allow_html=True)
        parent_messages = [msg for msg in conversation_history if msg['role'] == 'parent']
        strategies_used = [msg.get('strategy_used', strategy) for msg in parent_messages]
        
        # Display strategy usage
        st.write("**Strategy Usage:**")
        for strategy_name, count in {s: strategies_used.count(s) for s in set(strategies_used)}.items():
            st.write(f"- {strategy_name}: {count} times")
            
    st.markdown("<h2 class='section-header'>Final Reflection</h2>", unsafe_allow_html=True)
    
    with st.form(key='end_simulation_form'):
        current_reflection = {}
        strategies_used = set(msg.get('strategy_used', strategy) for msg in conversation_history if msg.get('role') == 'parent')
        
        for strategy_used in strategies_used:
            current_reflection[f"How effective was {strategy_used} in this conversation?"] = st.text_area(
                f"How effective was {strategy_used} in this conversation?",
                height=100,
                key=f"reflection_strategy_{strategy_used}"
            )
        
        current_reflection["What did you learn about your child's perspective?"] = st.text_area(
            "What did you learn about your child's perspective?",
            height=100,
            key="reflection_perspective"
        )
        
        current_reflection["What would you do differently next time?"] = st.text_area(
            "What would you do differently next time?",
            height=100,
            key="reflection_improvements"
        )
        
        submit_button = st.form_submit_button("Save Reflection", use_container_width=True)
    
    if submit_button:
        if not any(answer.strip() for answer in current_reflection.values()):
            st.warning("Please fill in at least one reflection question before saving.")
            return
            
        user_id = st.session_state.get('parent_name', 'Anonymous')
        if user_id == 'Anonymous':
            st.warning("Please enter your name in the sidebar to save reflections.")
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
        
        update_langsmith_run(
            st.session_state['run_id'],
            {"final_reflection": reflection_data}
        )
        
        success = save_reflection(user_id, 'end_simulation', reflection_data)
        
        if success:
            st.success("✨ Reflection saved successfully! View it in the 'View Reflections' tab.")
            if st.button("Start New Conversation", use_container_width=True):
                reset_simulation()
                st.rerun()
        else:
            st.error("Failed to save reflection. Please try again.")

def reset_simulation():
    st.session_state['conversation_history'] = []
    st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
    st.session_state['turn_count'] = 0
    st.session_state['strategy'] = "Active Listening"
    st.session_state['simulation_ended'] = False
    st.session_state['simulation_id'] = random.randint(1000, 9999)
    st.session_state['stored_responses'].clear()
    st.session_state['run_id'] = None

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
                    },
                    fallback_id=str(random.randint(1000, 9999))
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

def display_saved_reflections(user_id):
    st.markdown("<h2 class='section-header'>Your Saved Reflections</h2>", unsafe_allow_html=True)
    
    if not user_id:
        st.warning("Please enter your name in the sidebar first.")
        return
        
    reflections = load_reflections(user_id)
    
    if not reflections:
        st.info("No reflections found. Complete a role-play simulation to save reflections.")
        return
    
    st.write(f"Found {len(reflections)} saved reflection(s)")
    
    for reflection in reflections:
        with st.expander(f"Reflection from {reflection.timestamp.strftime('%Y-%m-%d %H:%M')}"):
            try:
                content = json.loads(reflection.content)
                if isinstance(content, dict):
                    if 'reflection_content' in content:
                        st.write("**Strategies used in this conversation:**")
                        st.write(", ".join(content.get('strategies_used', [])))
                        st.write("\n**Your Reflections:**")
                        for question, answer in content['reflection_content'].items():
                            if answer and answer.strip():
                                st.markdown(f"""
                                    <div class='reflection-item'>
                                        <strong>{question}</strong><br>
                                        {answer}
                                    </div>
                                    <hr>
                                """, unsafe_allow_html=True)
                    else:
                        for question, answer in content.items():
                            if answer and answer.strip():
                                st.markdown(f"""
                                    <div class='reflection-item'>
                                        <strong>{question}</strong><br>
                                        {answer}
                                    </div>
                                    <hr>
                                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying reflection: {str(e)}")

def main():
    if 'info_submitted' not in st.session_state:
        st.session_state['info_submitted'] = False
        
    if not st.session_state['info_submitted']:
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
        
        st.markdown("<h1 class='main-header'>Parenting Support Bot</h1>", unsafe_allow_html=True)
        
        # Component descriptions
        component_descriptions = {
            "Advice": "Get expert guidance on handling specific parenting situations based on evidence-based strategies.",
            "Conversation Starters": "Receive help initiating difficult conversations with suggested opening phrases and questions.",
            "Communication Techniques": "Discover helpful ways to talk with your child and get tips on how to give clear and simple answers.",
            "Role-Play Simulation": "Practice conversations in a safe environment to develop and refine your communication approach.",
            "View Reflections": "Review your saved insights and learning from previous practice sessions."
        }

        selected = st.radio(
            "Choose an option:",
            list(component_descriptions.keys()),
            horizontal=True,
            help="Select a tool that best matches your current needs"
        )

        # Display component description
        st.info(component_descriptions[selected])

        if selected == "Advice":
            display_advice(st.session_state['parent_name'], st.session_state['child_age'], st.session_state['situation'])
        elif selected == "Conversation Starters":
            display_conversation_starters(st.session_state['situation'])
        elif selected == "Communication Techniques":
            display_communication_techniques(st.session_state['situation'])
        elif selected == "Role-Play Simulation":
            simulate_conversation_streamlit(st.session_state['parent_name'], st.session_state['child_age'], st.session_state['situation'])
        elif selected == "View Reflections":
            display_saved_reflections(st.session_state['parent_name'])

# Add additional CSS for reflection display
st.markdown("""
    <style>
    .info-section {
        background-color: #f3f4f6;
        padding: 1.2em;
        border-radius: 0.8em;
        margin: 1em 0;
        border: 1px solid #60a5fa;
    }
    
    .reflection-item {
        margin: 1.2em 0;
        line-height: 1.8;
        padding: 1em;
        background-color: #f3f4f6;
        border-left: 4px solid #2563eb;
        border-radius: 0.5em;
    }
    
    .stRadio > div {
        display: flex;
        gap: 1em;
        flex-wrap: wrap;
    }
    
    .stRadio > div > label {
        padding: 0.8em 1.2em;
        border-radius: 0.5em;
        background-color: #f3f4f6;
        flex: 1;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .stRadio > div > label:hover {
        background-color: #60a5fa;
        color: white;
    }
    
    /* Textarea styling */
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
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
