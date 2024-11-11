import streamlit as st
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

# Strategy explanations
STRATEGY_EXPLANATIONS = {
    "Active Listening": "👂 Active Listening involves fully focusing on, understanding, and remembering what your child is saying. This helps them feel heard and valued.",
    "Positive Reinforcement": "⭐ Positive Reinforcement involves encouraging desired behaviors through specific praise or rewards, helping build self-esteem and motivation.",
    "Reflective Questioning": "❓ Reflective Questioning uses open-ended questions to help children think deeper and express themselves. For example: 'What do you think about...?'"
}

# Age-specific responses dictionary
AGE_SPECIFIC_RESPONSES = {
    "3-5 years": {
        "cooperative": [
            "*looking up with big eyes* Okay, I'll try...",
            "Can you help me?",
            "*nodding* I want to be good!",
            "Like this, Mommy/Daddy?",
            "*trying but struggling* It's hard!"
        ],
        "defiant": [
            "No! No! NO!",
            "*throwing self on floor* I DON'T WANNA!",
            "*covering ears* La la la, can't hear you!",
            "You're not the boss of me!",
            "*crying loudly* I want MY way!"
        ],
        "distracted": [
            "Look, my toy is dancing!",
            "Can I have a snack?",
            "*spinning around* Wheeeee!",
            "But I wanna play with my blocks!",
            "Is it time for cartoons?"
        ]
    },
    "6-9 years": {
        "cooperative": [
            "I'll clean up after I finish this part.",
            "Sorry, I didn't mean to...",
            "Can we make a deal?",
            "I promise I'll do better.",
            "Will you show me how?"
        ],
        "defiant": [
            "But that's not fair! Jamie never has to!",
            "*arms crossed* You can't make me!",
            "I hate these rules!",
            "You never let me do anything fun!",
            "Well, Sarah's parents let her!"
        ],
        "distracted": [
            "But first can I just...",
            "Wait, I forgot to tell you about...",
            "*staring out window* What's that bird doing?",
            "Can we do it later? I'm almost finished with...",
            "Oh! I just remembered something!"
        ]
    },
    "10-12 years": {
        "cooperative": [
            "Fine, I get it. Just give me a minute.",
            "I know, I know. I'm going.",
            "Okay, but can we talk about it first?",
            "I understand, but...",
            "I'll do it, just let me finish this."
        ],
        "defiant": [
            "*rolling eyes* Whatever.",
            "This is so unfair! You never understand!",
            "Everyone else gets to!",
            "*slamming door* Leave me alone!",
            "You're ruining everything!"
        ],
        "distracted": [
            "Yeah, just one more level...",
            "Hold on, I'm texting...",
            "In a minute... I'm doing something.",
            "Did you see what happened at school today?",
            "But I'm in the middle of something!"
        ]
    }
}

# Database setup
DATABASE_URL = "sqlite:///parenting_app.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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
        db_reflection = Reflection(
            user_id=user_id,
            type=reflection_type,
            content=json.dumps(content),
            langsmith_run_id=st.session_state.get('run_id'),
            timestamp=datetime.utcnow()
        )
        db.add(db_reflection)
        db.commit()
        db.refresh(db_reflection)
        print(f"Saved reflection for user {user_id}: {reflection_type}")
        return True
    except Exception as e:
        print(f"Error saving reflection: {str(e)}")
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

        Parent's response to react to: {parent_response}"""}
    ]

    # Add LangChain run tracking
    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=40
        )
        response = completion.choices[0].message.content.strip()
        
        # Store run ID for reflection linking
        if 'run_id' not in st.session_state:
            st.session_state['run_id'] = smith_client.create_run(
                run_type="chain",  # Added required parameter
                name="parenting_conversation",
                inputs={
                    "child_age": child_age,
                    "situation": situation,
                    "mood": mood,
                    "strategy": strategy
                }
            ).id
        
        # Log the interaction
        smith_client.update_run(
            st.session_state['run_id'],
            outputs={
                "parent_response": parent_response,
                "child_response": response,
                "strategy_used": strategy,
                "mood": mood
            }
        )
        
        st.session_state['stored_responses'][response_key] = response
        return response
    except Exception as e:
        print(f"Error in LangChain tracking: {e}")
        return st.session_state['stored_responses'].get(response_key, "I don't know what to say...")

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
@traceable(name="simulate_conversation")
def simulate_conversation_streamlit(name, child_age, situation):
    name = st.session_state.get('parent_name', name)
    child_name = st.session_state.get('child_name', '')
    child_age = st.session_state.get('child_age', child_age)
    situation = st.session_state.get('situation', situation)
    
    st.subheader("Parent-Child Role-Play Simulator")
    
    # Add introduction
    st.markdown("""
    Welcome to the conversation simulator! Here you can practice different communication strategies 
    with your child in a safe environment. Start by responding to the situation as you normally would, 
    and then try incorporating different communication strategies to see how they might change the interaction.
    
    **Your situation:** {}
    
    Begin by typing how you would typically respond to this situation.
    """.format(situation))

    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
        st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
        st.session_state['turn_count'] = 0
        st.session_state['strategy'] = "Active Listening"
        st.session_state['simulation_id'] = random.randint(1000, 9999)

        # Create initial LangChain run with required run_type
        try:
            st.session_state['run_id'] = smith_client.create_run(
                run_type="chain",
                name="parenting_conversation",
                inputs={
                    "parent_name": name,
                    "child_name": child_name,
                    "child_age": child_age,
                    "situation": situation,
                    "initial_strategy": "Active Listening"
                }
            ).id
        except Exception as e:
            print(f"Error creating initial LangChain run: {e}")

    # Display strategy selection using more interactive buttons
    st.write("Choose your communication strategy:")
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
                try:
                    smith_client.update_run(
                        st.session_state['run_id'],
                        outputs={"strategy_change": strategy}
                    )
                except Exception as e:
                    print(f"Error logging strategy change: {e}")
                st.rerun()
    
    # Display current strategy explanation
    st.info(STRATEGY_EXPLANATIONS[st.session_state['strategy']])

    # Display conversation history with improved formatting
    st.write("Conversation:")
    for msg in st.session_state['conversation_history']:
        col1, col2 = st.columns([8, 4])
        with col1:
            if msg['role'] == 'parent':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**{child_name}:** {msg['content']}")
        with col2:
            if msg['role'] == 'parent' and 'feedback' in msg:
                st.info(f"💡 {msg['feedback']}")

    # Parent's input section
    with st.form(key=f'parent_input_form_{st.session_state["simulation_id"]}_{st.session_state["turn_count"]}'):
        user_input = st.text_area(
            "Your response:",
            key=f"parent_input_{st.session_state['simulation_id']}_{st.session_state['turn_count']}",
            height=100
        )
        col1, col2 = st.columns(2)
        with col1:
            send_button = st.form_submit_button("Send Response", use_container_width=True)
        with col2:
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
        
        try:
            # Log parent's response to LangChain
            smith_client.update_run(
                st.session_state['run_id'],
                outputs={
                    f"turn_{st.session_state['turn_count']}_parent": {
                        "content": user_input,
                        "strategy": st.session_state['strategy'],
                        "feedback": feedback
                    }
                }
            )
        except Exception as e:
            print(f"Error logging parent response: {e}")
        
        # Generate child's response
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
            try:
                smith_client.update_run(
                    st.session_state['run_id'],
                    outputs={"mood_change": st.session_state['child_mood']}
                )
            except Exception as e:
                print(f"Error logging mood change: {e}")
        
        st.session_state['turn_count'] += 1
        st.rerun()
    
    if end_button:
        try:
            smith_client.update_run(
                st.session_state['run_id'],
                outputs={"conversation_ended": True, "total_turns": st.session_state['turn_count']}
            )
        except Exception as e:
            print(f"Error logging conversation end: {e}")
        end_simulation(st.session_state['conversation_history'], child_age, st.session_state['strategy'])

def end_simulation(conversation_history, child_age, strategy):
    st.session_state['simulation_ended'] = True
    st.write("The simulation has ended.")
    st.subheader("Final Reflection")
    
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
        
        submit_button = st.form_submit_button("Save Reflection")
    
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
        
        try:
            # Log reflection to LangChain
            smith_client.update_run(
                st.session_state['run_id'],
                outputs={"final_reflection": reflection_data}
            )
        except Exception as e:
            print(f"Error logging reflection to LangChain: {e}")
        
        success = save_reflection(user_id, 'end_simulation', reflection_data)
        
        if success:
            st.success("✨ Reflection saved successfully! View it in the 'View Reflections' tab.")
            if st.button("Start New Conversation"):
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
    st.session_state['run_id'] = None  # Reset LangChain run ID
def main():
    st.set_page_config(layout="wide", page_title="Parenting Support Bot")
    
    with st.sidebar:
        st.subheader("Parent Information")
        
        with st.form(key='parent_info_form'):
            parent_name = st.text_input("Your Name")
            child_name = st.text_input("Child's Name")
            
            age_ranges = ["3-5 years", "6-9 years", "10-12 years"]
            child_age = st.selectbox("Child's Age Range", age_ranges)
            
            situation = st.text_area("Describe the situation")
            submit_button = st.form_submit_button("Save Information")
        
        if submit_button:
            st.session_state['parent_name'] = parent_name
            st.session_state['child_name'] = child_name
            st.session_state['child_age'] = child_age
            st.session_state['situation'] = situation
            st.success("Information saved!")
            if 'conversation_history' in st.session_state:
                st.session_state.pop('conversation_history')
            st.rerun()
    
    st.title("Parenting Support Bot")

    selected = st.radio(
        "Choose an option:",
        ["Advice", "Conversation Starters", "Communication Techniques", "Role-Play Simulation", "View Reflections"],
        horizontal=True
    )

    parent_name = st.session_state.get('parent_name', '')
    child_name = st.session_state.get('child_name', '')
    child_age = st.session_state.get('child_age', '3-5 years')
    situation = st.session_state.get('situation', '')

    if selected == "Advice":
        display_advice(parent_name, child_age, situation)
    elif selected == "Conversation Starters":
        display_conversation_starters(situation)
    elif selected == "Communication Techniques":
        display_communication_techniques(situation)
    elif selected == "Role-Play Simulation":
        if not situation:
            st.warning("Please describe the situation in the sidebar before starting the simulation.")
        else:
            simulate_conversation_streamlit(parent_name, child_age, situation)
    elif selected == "View Reflections":
        display_saved_reflections(parent_name)

def display_saved_reflections(user_id):
    st.subheader("Your Saved Reflections")
    
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
                                st.markdown(f"**{question}**")
                                st.write(answer)
                                st.markdown("---")
                    else:
                        for question, answer in content.items():
                            if answer and answer.strip():
                                st.markdown(f"**{question}**")
                                st.write(answer)
                                st.markdown("---")
            except Exception as e:
                st.error(f"Error displaying reflection: {str(e)}")

@traceable(name="display_advice")
def display_advice(parent_name, child_age, situation):
    st.subheader("Parenting Advice")
    if situation:
        user_input = f"Parent: {parent_name}\nChild's age: {child_age}\nSituation: {situation}\nGoal: Get advice"
        try:
            with st.spinner('Processing your request...'):
                messages = [
                    {"role": "system", "content": "You are a parenting expert providing advice based on research-backed strategies."},
                    {"role": "user", "content": user_input}
                ]
                completion = openai.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )

                # Log advice request to LangChain
                run_id = smith_client.create_run(
                    run_type="chain",
                    name="parenting_advice",
                    inputs={
                        "parent_name": parent_name,
                        "child_age": child_age,
                        "situation": situation
                    }
                ).id
                
                smith_client.update_run(
                    run_id,
                    outputs={"advice": completion.choices[0].message.content}
                )
                
                st.markdown(completion.choices[0].message.content)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please describe the situation in the sidebar to get advice.")

@traceable(name="display_conversation_starters")
def display_conversation_starters(situation):
    st.subheader("Conversation Starters")
    if situation:
        try:
            with st.spinner('Generating conversation starters...'):
                # Create LangChain run first
                run_id = smith_client.create_run(
                    run_type="chain",
                    name="conversation_starters",
                    inputs={"situation": situation}
                ).id
                
                starters = generate_conversation_starters(situation)
                
                # Update run with results
                smith_client.update_run(
                    run_id,
                    outputs={"starters": starters}
                )
                
                st.write(starters)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please describe the situation in the sidebar to get conversation starters.")

@traceable(name="display_communication_techniques")
def display_communication_techniques(situation):
    st.subheader("Communication Techniques")
    if situation:
        try:
            with st.spinner('Generating communication techniques...'):
                # Create LangChain run first
                run_id = smith_client.create_run(
                    run_type="chain",
                    name="communication_techniques",
                    inputs={"situation": situation}
                ).id
                
                messages = [
                    {"role": "system", "content": "You are a parenting expert focused on effective communication strategies."},
                    {"role": "user", "content": f"Provide specific communication techniques for this situation: {situation}"}
                ]
                completion = openai.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )
                
                # Update run with results
                smith_client.update_run(
                    run_id,
                    outputs={"techniques": completion.choices[0].message.content}
                )
                
                st.markdown(completion.choices[0].message.content)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please describe the situation in the sidebar to get communication techniques.")

if __name__ == "__main__":
    main()
