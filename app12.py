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
os.environ["LANGCHAIN_TRACING_V2"] = 'true'

# Initialize LangSmith Client
smith_client = Client()

# Strategy explanations
STRATEGY_EXPLANATIONS = {
    "Active Listening": "Active Listening involves fully focusing on, understanding, and remembering what your child is saying. This helps them feel heard and valued.",
    "I-Messages": "I-Messages allow you to express your feelings and needs without blame or criticism. For example: 'I feel worried when...' instead of 'You always...'",
    "Positive Reinforcement": "Positive Reinforcement involves encouraging desired behaviors through specific praise or rewards, helping build self-esteem and motivation.",
    "Reflective Questioning": "Reflective Questioning uses open-ended questions to help children think deeper and express themselves. For example: 'What do you think about...?'"
}

# Define reflection questions
REFLECTION_QUESTIONS = [
    "How effective was the strategy you used in this interaction?",
    "What did you learn about your child's perspective?",
    "What would you do differently next time?",
]

# Database setup
DATABASE_URL = "sqlite:///parenting_app.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define Reflection model
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
    try:
        db = SessionLocal()
        db_reflection = Reflection(
            user_id=user_id,
            type=reflection_type,
            content=json.dumps(content),
            langsmith_run_id=st.session_state.get('run_id')
        )
        db.add(db_reflection)
        db.commit()
        db.refresh(db_reflection)
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
        "I-Messages": [
            "Good use of 'I feel...' - this helps express your emotions clearly.",
            "Try adding why you feel this way: 'I feel... when...'",
            "Consider explaining the impact: 'I feel... when... because...'"
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

def generate_child_response(conversation_history, child_age, situation, mood, strategy, parent_response):
    response_key = f"{parent_response}_{child_age}_{mood}_{strategy}"
    
    if response_key in st.session_state['stored_responses']:
        return st.session_state['stored_responses'][response_key]
    
    messages = [
        {"role": "system", "content": f"""You are simulating a {child_age}-year-old child responding to their parent. 
        The child is currently in a {mood} mood. Adapt your language, emotional responses, and cognitive abilities to match a typical {child_age}-year-old. 
        Consider the following situation: {situation}
        The parent is using the {strategy} communication strategy.

        {SIMULATION_CITATIONS}

        Remember to:
        1. Use age-appropriate language and vocabulary.
        2. Express emotions in a way typical for a {child_age}-year-old.
        3. Show the level of understanding and reasoning expected at this age.
        4. Display behaviors or responses that might challenge the parent, especially if in a defiant or distracted mood.
        5. Be consistent with the initial situation described and the current mood.
        6. Respond in a way that allows the parent to practice the {strategy} strategy.
        7. Consider the parent's most recent response: {parent_response}

        Respond naturally as a child would in the given situation and mood.
        """},
    ] + [{"role": "user" if msg["role"] == "parent" else "assistant", "content": msg["content"]} for msg in conversation_history]

    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150
    )
    response = completion.choices[0].message.content.strip()
    
    st.session_state['stored_responses'][response_key] = response
    return response
def simulate_conversation_streamlit(name, child_age, situation):
    name = st.session_state.get('parent_name', name)
    child_name = st.session_state.get('child_name', '')
    child_age = st.session_state.get('child_age', child_age)
    situation = st.session_state.get('situation', situation)
    
    st.subheader("Parent-Child Role-Play Simulator")
    
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
        st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
        st.session_state['turn_count'] = 0
        st.session_state['input_key'] = 0
        st.session_state['strategy'] = "Active Listening"
        st.session_state['simulation_id'] = random.randint(1000, 9999)
        st.session_state['changing_strategy'] = False

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Current Strategy: {st.session_state['strategy']}")
        st.markdown(f"*{STRATEGY_EXPLANATIONS[st.session_state['strategy']]}*")
    with col2:
        if st.button("Change Strategy", key=f"change_strategy_{st.session_state['simulation_id']}"):
            st.session_state['changing_strategy'] = True

    if st.session_state.get('changing_strategy', False):
        new_strategy = st.selectbox(
            "Choose a new communication strategy:",
            list(STRATEGY_EXPLANATIONS.keys()),
            index=list(STRATEGY_EXPLANATIONS.keys()).index(st.session_state['strategy']),
            key=f"strategy_select_{st.session_state['simulation_id']}"
        )
        st.info(STRATEGY_EXPLANATIONS[new_strategy])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Change", key=f"confirm_change_{st.session_state['simulation_id']}"):
                if new_strategy != st.session_state['strategy']:
                    st.session_state['strategy'] = new_strategy
                    st.success(f"Strategy changed to: {new_strategy}")
                st.session_state['changing_strategy'] = False
                st.rerun()
        with col2:
            if st.button("Cancel", key=f"cancel_change_{st.session_state['simulation_id']}"):
                st.session_state['changing_strategy'] = False
                st.rerun()

    tab1, tab2 = st.tabs(["Conversation", "Your Reflections"])

    with tab1:
        st.write("Conversation History:")
        for msg in st.session_state['conversation_history']:
            if msg['role'] == 'parent':
                st.markdown(f"**You:** {msg['content']}")
                if 'feedback' in msg:
                    st.info(f"Feedback: {msg['feedback']}")
            else:
                st.markdown(f"**{child_name}:** {msg['content']}")

        if len(st.session_state['conversation_history']) % 2 == 0:
            with st.form(key=f'parent_input_form_{st.session_state["simulation_id"]}_{st.session_state["turn_count"]}'):
                user_input = st.text_area("Your response:", key=f"parent_input_{st.session_state['simulation_id']}_{st.session_state['turn_count']}", height=100)
                col1, col2 = st.columns(2)
                with col1:
                    send_button = st.form_submit_button("Send Response")
                with col2:
                    pause_button = st.form_submit_button("Pause and Reflect")
            
            if send_button and user_input:
                feedback = provide_realtime_feedback(user_input, st.session_state['strategy'])
                st.session_state['conversation_history'].append({
                    "role": "parent", 
                    "content": user_input, 
                    "id": len(st.session_state['conversation_history']), 
                    "feedback": feedback
                })
                st.session_state['turn_count'] += 1
                st.rerun()
            
            if pause_button:
                with st.form(key=f'pause_reflect_form_{st.session_state["simulation_id"]}_{st.session_state["turn_count"]}'):
                    reflection = st.text_area("Jot down your thoughts or strategies:", key=f"reflection_{st.session_state['simulation_id']}_{st.session_state['turn_count']}")
                    submit_reflection = st.form_submit_button("Save Reflection")
        
                if submit_reflection and reflection.strip():
                    save_reflection(name, 'pause', {'content': reflection.strip()})
                    st.success("Reflection saved. Continuing simulation.")
                    st.rerun()

        else:
            try:
                parent_response = st.session_state['conversation_history'][-1]['content'] if st.session_state['conversation_history'] else ""
                
                child_response = generate_child_response(
                    st.session_state['conversation_history'], 
                    child_age, 
                    situation, 
                    st.session_state['child_mood'], 
                    st.session_state['strategy'], 
                    parent_response
                )
                st.session_state['conversation_history'].append({
                    "role": "child", 
                    "content": child_response, 
                    "id": len(st.session_state['conversation_history'])
                })
                
                if random.random() < 0.3:  # 30% chance to change mood
                    st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
                
                st.session_state['turn_count'] += 1
                st.rerun()
            except Exception as e:
                print(f"Error in child response generation: {e}")

        if len(st.session_state['conversation_history']) > 0:
            st.write("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("End Simulation", key=f"end_button_{st.session_state['simulation_id']}_{st.session_state['turn_count']}"):
                    end_simulation(st.session_state['conversation_history'], child_age, st.session_state['strategy'])
            with col2:
                if st.button("Start New Conversation", key=f"new_conversation_{st.session_state['simulation_id']}_{st.session_state['turn_count']}"):
                    reset_simulation()
                    st.rerun()
            
            if not st.session_state.get('simulation_ended', False):
                st.info("Once you end the simulation, you'll be prompted to reflect on the role-play and assess what you've learned.")

    with tab2:
        display_reflections(name if name else "Anonymous")

def display_reflections(user_id):
    st.subheader("Your Reflections")
    
    try:
        reflections = load_reflections(user_id)
        
        pause_reflections = [r for r in reflections if r.type == 'pause']
        end_simulation_reflections = [r for r in reflections if r.type == 'end_simulation']
        
        if pause_reflections:
            st.write("Pause and Reflect Entries:")
            for i, reflection in enumerate(pause_reflections[:5], 1):
                content = json.loads(reflection.content)
                with st.expander(f"Reflection {i} - {reflection.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                    st.text_area(
                        "Your thoughts",
                        content['content'],
                        height=100,
                        key=f"saved_pause_reflection_{reflection.id}",
                        disabled=True
                    )
            if len(pause_reflections) > 5:
                st.info(f"Showing the last 5 of {len(pause_reflections)} pause reflections.")
        else:
            st.info("No pause reflections recorded yet.")

        if end_simulation_reflections:
            st.write("End-of-Simulation Reflections:")
            for reflection in end_simulation_reflections[:3]:
                with st.expander(f"Simulation Reflection - {reflection.timestamp.strftime('%Y-%m-%d %H:%M')}"):
                    content = json.loads(reflection.content)
                    for question, answer in content.items():
                        if answer.strip():
                            st.text_area(
                                question,
                                answer,
                                height=100,
                                key=f"saved_answer_{reflection.id}_{question}",
                                disabled=True
                            )
            if len(end_simulation_reflections) > 3:
                st.info(f"Showing the last 3 of {len(end_simulation_reflections)} end-of-simulation reflections.")
        else:
            st.info("No end-of-simulation reflections recorded yet.")
    except Exception as e:
        st.error(f"An error occurred while loading reflections: {str(e)}")
def main():
    st.set_page_config(layout="wide", page_title="Parenting Support Bot")
    
    with st.sidebar:
        st.subheader("Parent Information")
        
        with st.form(key='parent_info_form'):
            parent_name = st.text_input("Your Name")
            child_name = st.text_input("Child's Name")
            
            # Updated age ranges
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
    
    st.title("Parenting Support Bot")

    selected = st.radio(
        "Choose an option:",
        ["Advice", "Conversation Starters", "Communication Techniques", "Role-Play Simulation"],
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
        simulate_conversation_streamlit(parent_name, child_age, situation)

def end_simulation(conversation_history, child_age, strategy):
    st.session_state['simulation_ended'] = True
    st.write("The simulation has ended.")
    st.subheader("Reflection")
    st.write("Take a moment to reflect on what you've learned from this role-play.")
    
    with st.form(key='end_simulation_form'):
        current_reflection = {}
        for i, question in enumerate(REFLECTION_QUESTIONS):
            formatted_question = question.replace("the strategy", f"the {strategy} strategy") if "the strategy" in question else question
            answer = st.text_area(formatted_question, height=100, key=f"reflection_{i}")
            current_reflection[formatted_question] = answer
        
        submit_button = st.form_submit_button("Save Reflection")
        
    if submit_button and any(answer.strip() for answer in current_reflection.values()):
        save_reflection(st.session_state.get('parent_name', 'Anonymous'), 'end_simulation', current_reflection)
        st.success("Reflection saved successfully.")
        
        feedback_message = f"""
        Thank you for completing this simulation using the {strategy} strategy. 
        Your reflections show thoughtful consideration of the interaction.
        
        Key takeaways to consider:
        - How the {strategy} techniques influenced your child's responses
        - Ways to adapt these techniques for different situations
        - Opportunities for growth in future interactions
        
        Would you like to try another scenario with a different strategy?
        """
        st.markdown(feedback_message)
        
        if st.button("Start New Simulation"):
            reset_simulation()
            st.rerun()

def reset_simulation():
    st.session_state['conversation_history'] = []
    st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
    st.session_state['turn_count'] = 0
    st.session_state['input_key'] = 0
    st.session_state['strategy'] = "Active Listening"
    st.session_state['simulation_ended'] = False
    st.session_state['simulation_id'] = random.randint(1000, 9999)
    st.session_state['stored_responses'].clear()

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
                st.markdown(completion.choices[0].message.content)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please describe the situation in the sidebar to get advice.")

def display_conversation_starters(situation):
    st.subheader("Conversation Starters")
    if situation:
        with st.spinner('Generating conversation starters...'):
            starters = generate_conversation_starters(situation)
        st.write(starters)
    else:
        st.warning("Please describe the situation in the sidebar to get conversation starters.")

def display_communication_techniques(situation):
    st.subheader("Communication Techniques")
    if situation:
        try:
            with st.spinner('Generating communication techniques...'):
                messages = [
                    {"role": "system", "content": "You are a parenting expert focused on effective communication strategies."},
                    {"role": "user", "content": f"Provide specific communication techniques for this situation: {situation}"}
                ]
                completion = openai.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )
                st.markdown(completion.choices[0].message.content)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please describe the situation in the sidebar to get communication techniques.")

if __name__ == "__main__":
    main()
