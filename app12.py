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

# Load environment - loads the OpenAI API key
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

# Prompt import
from lc_prompts import *
from lc_simulation_scenario_prompts import *

DEBUG = False

st.set_page_config(layout="wide", page_title="Parenting Support Bot")

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

Base.metadata.create_all(bind=engine)

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

# Call this function at the start of your script
setup_database()

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

msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

adaptation = False
if adaptation: 
    adaptation_msgs = StreamlitChatMessageHistory(key="adaptation_messages")
    adaptation_memory = ConversationBufferMemory(memory_key="adaptation_history", chat_memory=adaptation_msgs)

if st.session_state['llm_model'] == "gpt-4":
    prompt_datacollection = prompt_datacollection_4o

# Define reflection questions globally
REFLECTION_QUESTIONS = [
    "How effective was the strategy you used in this interaction?",
    "What did you learn about your child's perspective?",
    "What would you do differently next time?",
]

# LangChain agent setup
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: list[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction or AgentFinish:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        match = re.search(r"Action: (.*?)[\n]*Action Input:[\s]*(.*)", llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

llm = OpenAI(temperature=0)

tools = [
    Tool(
        name="Parenting Advice",
        func=lambda x: "This is where you'd put your parenting advice logic",
        description="useful for when you need to provide parenting advice"
    ),
]

prompt = CustomPromptTemplate(
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}""",
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

output_parser = CustomOutputParser()
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=[tool.name for tool in tools]
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Helper functions
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
        print(f"Reflection saved: {db_reflection.id}")
        return True
    except Exception as e:
        print(f"Error saving reflection: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)

# Call this at the start of your script
init_db()

def load_reflections(user_id):
    db = SessionLocal()
    try:
        print(f"Attempting to load reflections for user: {user_id}")
        reflections = db.query(Reflection).filter(Reflection.user_id == user_id).order_by(Reflection.timestamp.desc()).all()
        print(f"Loaded reflections: {[(r.id, r.type, r.content) for r in reflections]}")
        return reflections
    except Exception as e:
        print(f"Error loading reflections: {str(e)}")
        return []
    finally:
        db.close()

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
    conversation_starters = completion.choices[0].message.content.strip()
    return conversation_starters

def teach_communication_strategies(situation):
    prompt = f"""
    SYSTEM
    Use the provided citations delimited by triple quotes to answer questions. If the answer cannot be found in the citations, write "I could not find an answer."
    USER
    
    Academic Citations:
    {COMMUNICATION_STRATEGIES_CITATIONS}

    Website Resources:
    {WEBSITE_CITATIONS}

    Question: Provide communication strategies for the following situation with a child: {situation}. Include reflective questioning and tips to uncover underlying assumptions.
    """
    
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    communication_strategies = completion.choices[0].message.content.strip()
    return communication_strategies

def generate_alternative_responses(strategy, conversation_history, child_age):
    prompt = f"""
    Based on the following conversation history with a {child_age}-year-old child and the {strategy} strategy, suggest 3 alternative ways the parent could respond:
    
    {json.dumps(conversation_history)}
    
    Focus on using {strategy} techniques in your suggestions. Provide responses that are:
    1. Age-appropriate for a {child_age}-year-old child
    2. Aligned with the {strategy} communication strategy
    3. Constructive and aimed at improving the parent-child interaction
    """
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful parenting assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    alternatives = completion.choices[0].message.content.strip().split("\n")
    return [alt.strip() for alt in alternatives if alt.strip()]

def provide_realtime_feedback(parent_response, strategy):
    feedback_prompts = {
        "Active Listening": ["Great use of active listening!", "Consider reflecting the child's feelings more."],
        "I-Messages": ["Good job using 'I' statements!", "Good job stating what you want", "Try to express your feelings more clearly.", "Clearly state and explain how you feel about what is happening"],
        "Positive Reinforcement": ["Excellent positive reinforcement!", "Look for more opportunities to praise specific behaviors."],
        "Reflective Questioning": ["Great open-ended question!", "Try to dig deeper into the child's perspective."]
    }
    return random.choice(feedback_prompts[strategy])

def prompt_emotion_identification():
    return random.choice([True, False])  # Randomly decide whether to prompt for emotion identification


SIMULATION_CITATIONS = f"""
{ACTIVE_LISTENING_CITATIONS}
{COMMUNICATION_STRATEGIES_CITATIONS}
{I_MESSAGES_CITATIONS}
{POSITIVE_REINFORCEMENT_CITATIONS}
{WEBSITE_CITATIONS}
{REFLECTIVE_QUESTIONING_CITATIONS}
"""
def generate_child_response(conversation_history, child_age, situation, mood, strategy, parent_response):
    messages = [
        {"role": "system", "content": f"""You are simulating a {child_age}-year-old child responding to their parent. 
        The child is currently in a {mood} mood. Adapt your language, emotional responses, and cognitive abilities to match a typical {child_age}-year-old. 
        Consider the following situation: {situation}
        The parent is using the {strategy} communication strategy.

        Use the following citations to inform your responses:
        
        {SIMULATION_CITATIONS}

        Remember to:
        1. Use age-appropriate language and vocabulary.
        2. Express emotions in a way typical for a {child_age}-year-old.
        3. Show the level of understanding and reasoning expected at this age.
        4. Display behaviors or responses that might challenge the parent, especially if in a defiant or distracted mood.
        5. Be consistent with the initial situation described and the current mood.
        6. Respond in a way that allows the parent to practice the {strategy} strategy.
        7. Consider the parent's most recent response: {parent_response}

        Respond naturally as a child would in the given situation and mood, considering the parent's communication strategy and most recent response.
        """},
    ] + [{"role": "user" if msg["role"] == "parent" else "assistant", "content": msg["content"]} for msg in conversation_history]

    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150
    )
    return completion.choices[0].message.content.strip()

def log_interaction(parent_message, child_response):
    print(f"Parent: {parent_message}")
    print(f"Child: {child_response}")
    
def simulate_conversation_streamlit(name, child_age, situation):
    # Retrieve saved information from session state
    name = st.session_state.get('parent_name', name)
    user_id = name if name else "Anonymous"
    child_age = st.session_state.get('child_age', child_age)
    situation = st.session_state.get('situation', situation)
    
    st.subheader("Parent-Child Role-Play Simulator")
    
    # Initialize session state variables
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
        st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
        st.session_state['turn_count'] = 0
        st.session_state['input_key'] = 0
        st.session_state['strategy'] = "Active Listening"
        st.session_state['simulation_id'] = random.randint(1000, 9999)
        st.session_state['changing_strategy'] = False

    conversation_history = st.session_state['conversation_history']
    simulation_id = st.session_state['simulation_id']

    # Display current strategy and allow changing it
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Current Strategy: {st.session_state['strategy']}")
    with col2:
        if st.button("Change Strategy", key=f"change_strategy_{simulation_id}"):
            st.session_state['changing_strategy'] = True

    if st.session_state.get('changing_strategy', False):
        new_strategy = st.selectbox(
            "Choose a new communication strategy:",
            ["Active Listening", "I-Messages", "Positive Reinforcement", "Reflective Questioning"],
            index=["Active Listening", "I-Messages", "Positive Reinforcement", "Reflective Questioning"].index(st.session_state['strategy']),
            key=f"strategy_select_{simulation_id}"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Change", key=f"confirm_change_{simulation_id}"):
                if new_strategy != st.session_state['strategy']:
                    st.session_state['strategy'] = new_strategy
                    st.success(f"Strategy changed to: {new_strategy}")
                st.session_state['changing_strategy'] = False
                st.rerun()
        with col2:
            if st.button("Cancel", key=f"cancel_change_{simulation_id}"):
                st.session_state['changing_strategy'] = False
                st.rerun()

    # Add tabs for conversation and reflections
    tab1, tab2 = st.tabs(["Conversation", "Your Reflections"])

    with tab1:
        # Chat-like interface
        st.write("Conversation History:")
        for msg in conversation_history:
            if msg['role'] == 'parent':
                st.markdown(f"**{name} (Parent):** {msg['content']}")
                if 'feedback' in msg:
                    st.info(f"Feedback: {msg['feedback']}")
            else:
                st.markdown(f"**Child:** {msg['content']}")

        # Parent's turn
        if len(conversation_history) % 2 == 0:
            with st.form(key=f'parent_input_form_{simulation_id}_{st.session_state["turn_count"]}'):
                user_input = st.text_area("Your response as the parent:", key=f"parent_input_{simulation_id}_{st.session_state['turn_count']}", height=100)
                col1, col2 = st.columns(2)
                with col1:
                    send_button = st.form_submit_button("Send Response")
                with col2:
                    pause_button = st.form_submit_button("Pause and Reflect")
            
            if send_button and user_input:
                feedback = provide_realtime_feedback(user_input, st.session_state['strategy'])
                conversation_history.append({"role": "parent", "content": user_input, "id": len(conversation_history), "feedback": feedback})
                st.session_state['turn_count'] += 1
                st.rerun()
            
            if pause_button:
                with st.form(key=f'pause_reflect_form_{simulation_id}_{st.session_state["turn_count"]}'):
                    reflection = st.text_area("Jot down your thoughts or strategies:", key=f"reflection_{simulation_id}_{st.session_state['turn_count']}")
                    submit_reflection = st.form_submit_button("Save Reflection")
        
                if submit_reflection and reflection.strip():
                    print(f"Attempting to save reflection: {name}, pause, {{'content': {reflection.strip()}}}")
                    save_result = save_reflection(name, 'pause', {'content': reflection.strip()})
                    print(f"Save reflection result: {save_result}")
                    if save_result:
                        st.success("Reflection saved. Continuing simulation.")
                    else:
                        st.error("Failed to save reflection. Please try again.")
                    st.experimental_rerun()

        # Child's turn (hidden from UI)
        else:
            try:
                parent_response = conversation_history[-1]['content'] if conversation_history else ""
                
                child_response = generate_child_response(
                    conversation_history, 
                    child_age, 
                    situation, 
                    st.session_state['child_mood'], 
                    st.session_state['strategy'], 
                    parent_response
                )
                conversation_history.append({"role": "child", "content": child_response, "id": len(conversation_history)})
                
                # Log the current state for debugging
                print(f"Conversation history length: {len(conversation_history)}")
                print(f"Last entry: {conversation_history[-1]}")
                
                # Only log if there's a previous interaction (i.e., at least 2 elements in the history)
                if len(conversation_history) >= 2:
                    log_interaction(conversation_history[-2]['content'], child_response)
                else:
                    print("Not enough history to log interaction")
                
                if prompt_emotion_identification():
                    emotion = st.radio("What emotion do you think your child is feeling right now?", ["Happy", "Sad", "Angry", "Scared", "Confused"], key=f"emotion_{simulation_id}_{st.session_state['turn_count']}")
                    st.write(f"You identified that your child might be feeling {emotion}. Keep this in mind as you respond.")
                
                if random.random() < 0.3:
                    st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
                
                st.session_state['turn_count'] += 1
                st.rerun()
            except IndexError as e:
                print(f"IndexError occurred: {e}")
                print(f"Current conversation history: {conversation_history}")
            except Exception as e:
                print(f"Unexpected error occurred: {e}")

        # Options after each turn
        if len(conversation_history) > 0:
            st.write("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("End Simulation", key=f"end_button_{simulation_id}_{st.session_state['turn_count']}"):
                    end_simulation(conversation_history, child_age, st.session_state['strategy'])
            with col2:
                if st.button("Start New Conversation", key=f"new_conversation_{simulation_id}_{st.session_state['turn_count']}"):
                    reset_simulation()
                    st.rerun()
            
            # Only show the info message if the simulation hasn't ended
            if not st.session_state.get('simulation_ended', False):
                st.info("Once you end the simulation, you'll be prompted to reflect on the role-play and assess what you've learned.")

    with tab2:
        display_reflections(name if name else "Anonymous")


def end_simulation(conversation_history, child_age, strategy):
    st.session_state['simulation_ended'] = True
    st.write("The simulation has ended.")
    st.subheader("Reflection")
    st.write("Consider what you've learned from this role-play.")
    
    with st.form(key='end_simulation_form'):
        current_reflection = {}
        for i, question in enumerate(REFLECTION_QUESTIONS):
            formatted_question = question.replace("the strategy", f"the {strategy} strategy") if "the strategy" in question else question
            answer = st.text_area(formatted_question, height=100, key=f"reflection_{i}")
            current_reflection[formatted_question] = answer
        
        submit_button = st.form_submit_button("Save End-of-Simulation Reflection")
        
    if submit_button and any(answer.strip() for answer in current_reflection.values()):
        save_reflection(st.session_state.get('parent_name', 'Anonymous'), 'end_simulation', current_reflection)
        st.success("End-of-simulation reflection saved.")
        st.rerun()

def reset_simulation():
    st.session_state['conversation_history'] = []
    st.session_state['child_mood'] = random.choice(['cooperative', 'defiant', 'distracted'])
    st.session_state['turn_count'] = 0
    st.session_state['input_key'] = 0
    st.session_state['strategy'] = "Active Listening"
    st.session_state['simulation_ended'] = False
    st.session_state['simulation_id'] = random.randint(1000, 9999)

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
                st.text_area(f"Pause Reflection {i} - {reflection.timestamp}", content['content'], height=100, key=f"saved_pause_reflection_{reflection.id}")
            if len(pause_reflections) > 5:
                st.info(f"Showing the last 5 of {len(pause_reflections)} pause reflections.")
        else:
            st.info("You haven't made any pause reflections yet.")

        if end_simulation_reflections:
            st.write("End-of-Simulation Reflections:")
            for reflection in end_simulation_reflections[:3]:
                st.write(f"Reflection from {reflection.timestamp}:")
                content = json.loads(reflection.content)
                for question, answer in content.items():
                    if answer.strip():
                        st.text_area(question, answer, height=100, key=f"saved_answer_{reflection.id}_{question}")
                st.write("---")
            if len(end_simulation_reflections) > 3:
                st.info(f"Showing the last 3 of {len(end_simulation_reflections)} end-of-simulation reflections.")
        else:
            st.info("You haven't completed any end-of-simulation reflections yet.")
    except Exception as e:
        st.error(f"An error occurred while displaying reflections: {str(e)}")
    
    print(f"Displaying reflections for user: {user_id}")
    print(f"Loaded reflections: {reflections}")

def main():
    # Sidebar for user info
    with st.sidebar:
        st.subheader("Parent Information")
        
        # Use a form for parent information
        with st.form(key='parent_info_form'):
            name = st.text_input("Your Name")
            child_age = st.number_input("Child's Age", min_value=4, max_value=10)
            situation = st.text_area("Describe the situation")
            
            # Add a submit button to the form
            submit_button = st.form_submit_button("Save Parent Information")
        
        if submit_button:
            # Save the information to session state
            st.session_state['parent_name'] = name
            st.session_state['child_age'] = child_age
            st.session_state['situation'] = situation
            st.success("Parent information saved!")
    
    # Main content area
    st.title("Parenting Support Bot")

    # Tabbed interface using st.radio
    selected = st.radio(
        "Choose an option:",
        ["Advice", "Conversation Starters", "Communication Techniques", "Role-Play Simulation"],
        horizontal=True
    )

    # Use the saved information from session state
    name = st.session_state.get('parent_name', '')
    child_age = st.session_state.get('child_age', 4)
    situation = st.session_state.get('situation', '')

    if selected == "Advice":
        display_advice(name, child_age, situation)
    elif selected == "Conversation Starters":
        display_conversation_starters(situation)
    elif selected == "Communication Techniques":
        display_communication_techniques(situation)
    elif selected == "Role-Play Simulation":
        simulate_conversation_streamlit(name, child_age, situation)

def display_advice(name, child_age, situation):
    st.subheader("Parenting Advice")
    if situation:
        user_input = f"Parent: {name}\nChild's age: {child_age}\nSituation: {situation}\nGoal: Get advice"
        try:
            with st.spinner('Processing your request...'):
                response = agent_executor.run(user_input)
            st.markdown(response)
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
        with st.spinner('Generating communication techniques...'):
            techniques = teach_communication_strategies(situation)
        st.write(techniques)
    else:
        st.warning("Please describe the situation in the sidebar to get communication techniques.")

if __name__ == "__main__":
    main()