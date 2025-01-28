import streamlit as st

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

/* Persona UI Components */
.persona-container {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.persona-header {
    margin-bottom: 16px;
}

.persona-header h3 {
    color: #4338CA;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 8px;
}

/* Button Styling */
div.stButton > button {
    height: 3em;
    font-size: 1em;
    font-weight: 500;
    background-color: #4338CA;
    color: white;
    border-radius: 0.5em;
    transition: all 0.2s ease;
    width: 100%;
    margin-bottom: 1rem;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    background-color: #4F46E5;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

div.stButton > button[kind="secondary"] {
    background-color: #E5E7EB;
    color: #374151;
}

div.stButton > button[kind="secondary"]:hover {
    background-color: #D1D5DB;
}

/* Form Elements */
.stTextInput input, .stTextArea textarea {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 12px;
    font-size: 1em;
    transition: border-color 0.2s;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #4338CA;
    box-shadow: 0 0 0 2px rgba(67, 56, 202, 0.1);
}

/* Tabs Styling */
.stTabs {
    background: #F3F4F6;
    padding: 8px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.stTab {
    background: white;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}

.stTab[aria-selected="true"] {
    background-color: #4338CA;
    color: white;
}

/* Checkbox and Radio Styling */
.stCheckbox, .stRadio {
    padding: 8px;
    border-radius: 6px;
    transition: background-color 0.2s;
}

.stCheckbox:hover, .stRadio:hover {
    background-color: #F3F4F6;
}

/* Selectbox Styling */
.stSelectbox [data-baseweb=select] {
    background-color: white;
    border: 2px solid #4338CA;
    border-radius: 8px;
    padding: 6px 12px;
}

.stSelectbox [data-baseweb=select]:hover {
    border-color: #4F46E5;
    box-shadow: 0 2px 4px rgba(99, 102, 241, 0.1);
}

/* Slider Styling */
.stSlider {
    padding: 10px 0;
}

.stSlider .stSlideHandle {
    background-color: #4338CA;
    border: 2px solid white;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Grid Layout */
.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin: 16px 0;
}

/* Form Section Styling */
.form-section {
    background: #F9FAFB;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 16px;
}

.form-section-header {
    font-weight: 600;
    color: #374151;
    margin-bottom: 12px;
}

/* Helper Text */
.helper-text {
    color: #6B7280;
    font-size: 0.875rem;
    margin-top: 4px;
}

/* Message Styling */
.message-parent, .message-child {
    padding: 1.2em;
    border-radius: 1em;
    margin: 0.8em 0;
    max-width: 80%;
}

.message-parent {
    background-color: #4338CA;
    color: white;
    margin-left: auto;
}

.message-child {
    background-color: #F3F4F6;
    color: #1A1A1A;
    margin-right: auto;
}

/* Save Profile Section */
.save-profile-section {
    border-top: 1px solid #E5E7EB;
    padding-top: 16px;
    margin-top: 24px;
}

/* Feedback Messages */
.stInfo, .stSuccess, .stWarning, .stError {
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .grid-container {
        grid-template-columns: 1fr;
    }
    
    .message-parent, .message-child {
        max-width: 90%;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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

    def save_parent_information(self, parent_data: dict) -> Tuple[bool, Optional[str]]:
        """Save parent information to Supabase
        
        Args:
            parent_data (dict): Dictionary containing parent information
                Required keys: prolific_id, child_name, child_age, situation
                
        Returns:
            Tuple[bool, Optional[str]]: Success status and error message if any
        """
        if not self.ensure_initialized():
            return False, "Failed to initialize Supabase connection"

        try:
            # Validate required fields
            required_fields = ['prolific_id', 'child_name', 'child_age', 'situation']
            missing_fields = [f for f in required_fields if f not in parent_data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            # Add timestamp
            parent_data['created_at'] = datetime.utcnow().isoformat()
            
            # Insert into parent_information table
            result = self.supabase.table('parent_information').insert(parent_data).execute()
            
            if not result.data:
                raise ValueError("No data returned from insert operation")

            return True, result.data[0].get('id')

        except Exception as e:
            error_msg = f"Error saving parent information: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg

    def get_parent_information(self, prolific_id: str) -> Tuple[bool, Optional[dict]]:
        """Retrieve parent information from Supabase
        
        Args:
            prolific_id (str): Prolific ID to look up
            
        Returns:
            Tuple[bool, Optional[dict]]: Success status and parent data if found
        """
        if not self.ensure_initialized():
            return False, None

        try:
            result = self.supabase.table('parent_information')\
                .select("*")\
                .eq('prolific_id', prolific_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()

            if result.data:
                return True, result.data[0]
            return True, None

        except Exception as e:
            print(f"Error retrieving parent information: {e}")
            traceback.print_exc()
            return False, None

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

    def save_item_to_supabase(self, parent_id: str, item_type: str, title: str, content: str, metadata: dict = None) -> Tuple[bool, Optional[str]]:
        """Save an item to the saved_content table
        
        Args:
            parent_id (str): Prolific ID of the parent
            item_type (str): Type of content ('advice', 'technique', 'starter')
            title (str): Title of the saved item
            content (str): Main content text
            metadata (dict, optional): Additional metadata as JSON
        
        Returns:
            Tuple[bool, Optional[str]]: Success status and error message if any
        """
        if not self.ensure_initialized():
            return False, "Failed to initialize Supabase connection"

        try:
            data = {
                "parent_id": parent_id,
                "item_type": item_type,
                "title": title,
                "content": content,
                "metadata": metadata or {},
                "saved_at": datetime.utcnow().isoformat()
            }

            result = self.supabase.table('saved_content').insert(data).execute()
            
            if not result.data:
                raise ValueError("No data returned from insert operation")

            return True, None

        except Exception as e:
            error_msg = f"Error saving content: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg

    def get_saved_items(self, parent_id: str, item_type: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """Retrieve saved items for a parent
        
        Args:
            parent_id (str): Prolific ID of the parent
            item_type (str, optional): Filter by specific type
            
        Returns:
            Tuple[bool, List[Dict[str, Any]]]: Success status and list of saved items
        """
        if not self.ensure_initialized():
            return False, []

        try:
            query = self.supabase.table('saved_content').select("*").eq('parent_id', parent_id)
            
            if item_type:
                query = query.eq('item_type', item_type)
                
            result = query.order('saved_at', desc=True).execute()
            
            return True, result.data if result.data else []

        except Exception as e:
            print(f"Error retrieving saved items: {e}")
            traceback.print_exc()
            return False, []

    def delete_saved_item(self, item_id: str) -> Tuple[bool, Optional[str]]:
        """Delete a saved item
        
        Args:
            item_id (str): ID of the item to delete
            
        Returns:
            Tuple[bool, Optional[str]]: Success status and error message if any
        """
        if not self.ensure_initialized():
            return False, "Failed to initialize Supabase connection"

        try:
            result = self.supabase.table('saved_content').delete().eq('id', item_id).execute()
            return True, None
        except Exception as e:
            error_msg = f"Error deleting saved item: {str(e)}"
            print(error_msg)
            return False, error_msg

class PersonaManager:
    def __init__(self, supabase_manager):
        self.supabase = supabase_manager
        
    def save_persona(self, parent_id: str, persona_data: dict) -> Tuple[bool, str]:
        try:
            if not self.supabase.ensure_initialized():
                return False, "Database not initialized"
                
            data = {
                'parent_id': parent_id,
                'persona_name': persona_data.get('name'),
                'persona_data': persona_data,
                'created_at': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.supabase.table('child_personas').insert(data).execute()
            return (True, "Persona saved successfully") if result.data else (False, "Failed to save persona")
            
        except Exception as e:
            print(f"Error saving persona: {e}")
            return False, str(e)
    
    def load_personas(self, parent_id: str) -> Tuple[bool, List[dict]]:
        try:
            result = self.supabase.supabase.table('child_personas')\
                .select("*")\
                .eq('parent_id', parent_id)\
                .execute()
                
            return (True, result.data) if result.data else (True, [])
            
        except Exception as e:
            print(f"Error loading personas: {e}")
            return False, []
    
    def update_persona(self, persona_id: str, updated_data: dict) -> Tuple[bool, str]:
        try:
            result = self.supabase.supabase.table('child_personas')\
                .update({"persona_data": updated_data})\
                .eq('id', persona_id)\
                .execute()
                
            return (True, "Persona updated successfully") if result.data else (False, "Failed to update persona")
            
        except Exception as e:
            print(f"Error updating persona: {e}")
            return False, str(e)
    
    def delete_persona(self, persona_id: str) -> Tuple[bool, str]:
        try:
            result = self.supabase.supabase.table('child_personas')\
                .delete()\
                .eq('id', persona_id)\
                .execute()
                
            return (True, "Persona deleted successfully") if result.data else (False, "Failed to delete persona")
            
        except Exception as e:
            print(f"Error deleting persona: {e}")
            return False, str(e)
        
# Initialize managers
supabase_manager = SupabaseManager()
persona_manager = PersonaManager(supabase_manager)

def display_persona_customization():
    """Display and handle the enhanced persona customization interface"""
    st.markdown("""
        <style>
        /* Persona Customization Styles */
        .persona-container {
            background-color: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .option-button {
            background-color: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 8px 16px;
            margin: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .option-button.selected {
            background-color: #2563eb;
            color: white;
            border-color: #2563eb;
        }
        
        .behavior-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
            margin: 12px 0;
        }
        
        .behavior-item {
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .behavior-item.selected {
            background-color: #bfdbfe;
            border-color: #3b82f6;
            color: #1e40af;
        }
        
        .form-section {
            background-color: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .section-title {
            color: #2563eb;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Button to show customization
    if not st.session_state.get('show_persona_customization', False):
        if st.button("Customize Response Style", type="primary", use_container_width=True):
            st.session_state['show_persona_customization'] = True
            st.rerun()
        return

    st.markdown("### Customize Child Persona")
    
    # Load saved profiles first
    st.markdown("#### Saved Child Profiles")
    saved_profiles = st.session_state.get('saved_personas', {})
    profile_options = ["(None)"] + list(saved_profiles.keys())
    selected_profile = st.selectbox(
        "Select a saved profile to load",
        options=profile_options,
        key="profile_selector"
    )

    # If a profile is selected, load it
    if selected_profile != "(None)" and selected_profile in saved_profiles:
        profile_data = saved_profiles[selected_profile]
        if 'child_persona' not in st.session_state:
            st.session_state['child_persona'] = {}
        st.session_state['child_persona'] = profile_data
        
    with st.container():
        st.markdown("### Communication Style")
        communication_style = st.text_area(
            "Describe how your child typically communicates",
            value=st.session_state.get('temp_communication_style', ''),
            placeholder="e.g., how they express themselves in everyday conversations, asks many 'why' questions, or uses short answers...",
            help="Write a few sentences about how they normally speak or express themselves",
            height=130
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Emotional Expression")
            emotion_options = ["Very Reserved", "Somewhat Reserved", "Balanced", 
                             "Somewhat Expressive", "Very Expressive"]
            emotion_style = st.radio(
                "How does your child typically show their emotions? Choose emotional expression level",
                emotion_options,
                index=emotion_options.index(st.session_state.get('temp_emotion_style', 'Balanced')),
                horizontal=True
            )

        with col2:
            st.markdown("#### Response Length")
            length_options = ["Very Brief", "Brief", "Medium", "Detailed", "Very Detailed"]
            response_length = st.radio(
                "How detailed are your child's typical responses? Choose typical response length",
                length_options,
                index=length_options.index(st.session_state.get('temp_response_length', 'Medium')),
                horizontal=True
            )

        st.markdown("#### Common Behaviors")
        behavior_options = [
            "Argues and debates frequently",
            "Becomes quiet when upset",
            "Physically expressive",
            "Uses humor or sarcasm",
            "Asks many questions",
            "Negotiates extensively",
            "Gets loud when excited",
            "Gives minimal responses",
            "Shows physical affection",
            "Changes subject often",
            "Takes time to process"
        ]
        
        selected_behaviors = st.multiselect(
            "Select all the behaviors that match your child's typical communication style. These help create more realistic responses in the simulation.",
            options=behavior_options,
            default=st.session_state.get('temp_behaviors', [])
        )

        # Profile saving section
        st.markdown("---")
        profile_name = st.text_input(
            "Profile Name",
            value=st.session_state.get('temp_profile_name', ''),
            placeholder="e.g., 'After School Mood', 'Weekend Chatty', etc."
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✕ Close", type="secondary", use_container_width=True):
                st.session_state['show_persona_customization'] = False
                st.rerun()

        with col2:
            if st.button("💾 Save Profile", type="primary", use_container_width=True):
                if not communication_style:
                    st.error("Please describe your child's communication style before saving.")
                elif not profile_name:
                    st.error("Please provide a name for this profile.")
                else:
                    new_persona = {
                        "type": "detailed",
                        "communication_style": communication_style,
                        "emotion_style": emotion_style,
                        "response_length": response_length,
                        "response_patterns": selected_behaviors,
                        "typical_phrases": communication_style,
                        "communication_preferences": selected_behaviors
                    }
                    
                    # Save to session state
                    if 'saved_personas' not in st.session_state:
                        st.session_state['saved_personas'] = {}
                    st.session_state['saved_personas'][profile_name] = new_persona
                    
                    # Update current persona
                    st.session_state['child_persona'] = new_persona
                    update_child_mood(new_persona)
                    
                    # Save to database
                    if hasattr(st.session_state, 'supabase_client'):
                        try:
                            success, result = supabase_manager.save_persona(
                                st.session_state['parent_name'],
                                {
                                    "name": profile_name,
                                    "data": new_persona
                                }
                            )
                            if success:
                                st.success(f"Profile '{profile_name}' saved successfully!")
                            else:
                                st.error(f"Failed to save profile: {result}")
                        except Exception as e:
                            st.error(f"Error saving profile: {str(e)}")
                    else:
                        st.success(f"Profile '{profile_name}' saved successfully!")
                    
                    # Clear temporary states
                    st.session_state['show_persona_customization'] = False
                    st.rerun()

    # Store current values in temporary session state
    st.session_state['temp_communication_style'] = communication_style
    st.session_state['temp_emotion_style'] = emotion_style
    st.session_state['temp_response_length'] = response_length
    st.session_state['temp_behaviors'] = selected_behaviors
    st.session_state['temp_profile_name'] = profile_name

def init_session_state():
    """Initialize all session state variables with proper defaults."""
    session_vars = {
        # Core session identifiers
        'run_id': str(uuid4()),
        'simulation_id': str(uuid4()),
        'current_simulation_id': None,
        
        # User information
        'info_submitted': False,
        'parent_name': None,
        'child_name': None,
        'child_age': None,
        'situation': None,
        
        # Conversation tracking
        'conversation_history': [],
        'turn_count': 0,
        'stored_responses': {},
        'simulation_ended': False,
        
        # Child persona - Initialize with default values
        'child_persona': {
            'type': 'detailed',
            'communication_style': '',
            'emotion_style': 'Balanced',
            'response_length': 'Medium',
            'response_patterns': [],
            'typical_phrases': '',
            'communication_preferences': []
        },
        
        # Child mood will be determined based on persona and context
        'child_mood': 'neutral',
        
        # UI state
        'show_persona_customization': False,
        'strategy': "Active Listening",  # Default strategy
        'show_tutorial': True,
        
        # Feature tracking
        'visited_features': set(),
        'feature_outputs': {
            'advice': {},
            'conversation_starters': {},
            'communication_techniques': {},
            'simulation_history': []
        },
        
        # Feature usage tracking
        'feature_usage': {
            'advice_views': 0,
            'conversation_starter_uses': 0,
            'technique_views': 0,
            'simulation_runs': 0
        },
        
        # User preferences and settings
        'consent': False,
        'exp_data': True,
        'llm_model': "gpt-4",
        
        # Parent information tracking
        'parent_info_id': None,
        'saved_personas': {},
        
        # Error tracking
        'last_error': None,
        'error_count': 0,
        
        # Strategy states
        'active_strategy': None,
        'strategy_feedback': {},
        
        # Conversation state tracking
        'last_parent_response': None,
        'last_child_response': None,
        'conversation_context': [],
        
        # Performance metrics
        'response_times': [],
        'strategy_usage': {},
        'feedback_given': [],
        
        # UI preferences
        'show_feedback': True,
        'show_hints': True,
        'compact_view': False,
        
        # Session analytics
        'session_start_time': datetime.now(),
        'last_interaction_time': datetime.now(),
        'session_duration': 0,
        'interaction_count': 0
    }
    
    # Initialize or update session state variables
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

    # Only determine child mood if we have complete context
    if all(st.session_state.get(k) for k in ['child_persona', 'situation', 'child_age']):
        try:
            # Determine mood dynamically based on current context
            mood = determine_child_mood(
                st.session_state['child_persona'],
                st.session_state['situation'],
                st.session_state['child_age']
            )
            st.session_state['child_mood'] = mood
        except Exception as e:
            print(f"Error determining mood: {e}")
            if 'child_mood' not in st.session_state:
                st.session_state['child_mood'] = 'neutral'

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

def generate_child_response(conversation_history, child_age, situation, mood, strategy, parent_response):
    """Enhanced child response generation with persona handling"""
    child_name = st.session_state.get('child_name', 'the child')
    response_key = f"{parent_response}_{child_age}_{mood}_{strategy}"
    
    if response_key in st.session_state['stored_responses']:
        return st.session_state['stored_responses'][response_key]
    
    recent_chat = ' | '.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    
    # Persona info
    persona = st.session_state.get('child_persona', {})
    behavior_profile = f"""
    Communication:
    - Typical phrases: {persona.get('typical_phrases', '')}
    - Response to 'no': {persona.get('response_to_no', '')}
    - Communication patterns: {', '.join(persona.get('communication_preferences', []))}
    - Expression level: {persona.get('emotion_style', 'Balanced')}
    - Response length: {persona.get('response_length', 'Medium')}
    """

    trigger_detected = next((trigger for trigger in persona.get('communication_preferences', []) 
                           if trigger.lower() in parent_response.lower()), None)

    messages = [
        {"role": "system", "content": f"""You are simulating {child_name}'s responses.
        Core Info:
        - Name: {child_name}
        - Age: {child_age}
        - Mood: {mood}
        - Situation: {situation}
        - Chat History: {recent_chat}
        
        {behavior_profile}
        
        {f'IMPORTANT - Triggered pattern detected: {trigger_detected}' if trigger_detected else ''}
        
        Response Guidelines:
        1. Use specified communication style and phrases
        2. Match emotional expression level
        3. Maintain age-appropriate responses
        4. Stay in character
        5. Keep responses concise (1-2 sentences)
        6. If triggered, show relevant emotional/behavioral pattern
        
        Parent's message: "{parent_response}" """},
        {"role": "user", "content": f"Generate {child_name}'s response:"}
    ]

    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=60,
            presence_penalty=0.6
        )
        response = completion.choices[0].message.content.strip()
        st.session_state['stored_responses'][response_key] = response
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"{child_name} doesn't respond."

def determine_child_mood(persona: dict, situation: str, child_age: str) -> str:
    """Determines mood based on persona, situation and age"""
    persona_weights = analyze_child_persona(persona)
    situation_weights = analyze_situation_context(situation, child_age)
    
    final_weights = {
        mood: (p_weight * 0.6) + (s_weight * 0.4)
        for mood, (p_weight, s_weight) in 
        zip(persona_weights.keys(), zip(persona_weights.values(), situation_weights.values()))
    }
    
    total = sum(final_weights.values())
    probabilities = {m: w/total for m, w in final_weights.items()}
    
    return random.choices(
        list(probabilities.keys()),
        weights=list(probabilities.values()),
        k=1
    )[0]

def display_advice(parent_name: str, child_age: str, situation: str):
    """Display parenting advice with save functionality, using a softer color palette and black text."""
    st.markdown("<h2 class='section-header'>Parenting Advice</h2>", unsafe_allow_html=True)
    
    if not situation:
        st.warning("Please describe the situation to get advice.")
        return

    try:
        with st.spinner('Generating advice...'):
            messages = [
                {
                    "role": "system",
                    "content": f"""
                        You are a parenting expert. Generate 4 specific, practical pieces of advice for the given situation.
                        Each piece of advice should be:
                        - Detailed enough to be actionable (25-30 words)
                        - Specific to the situation and child's age
                        - Based on evidence-backed parenting strategies
                        - Include both what to do and why it works

                        Format each piece of advice as a JSON array of 4 objects with these fields:
                        - title: A short 2-3 word summary
                        - advice: The detailed advice
                        - icon: A relevant emoji
                        - color: The background color code (use exactly these colors in order):
                          ["#D9E9FF", "#C2FAD8", "#FFECC2", "#E8D9FF"]

                        Return exactly 4 items.
                    """
                },
                {
                    "role": "user",
                    "content": f"Child age: {child_age}\nSituation: {situation}\nGenerate specific advice for handling this situation."
                }
            ]

            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            advice_list = json.loads(completion.choices[0].message.content)
            
            # --- Updated CSS to remove forced white text and rely on the inline style we add below. ---
            st.markdown("""
                <style>
                    .advice-card {
                        padding: 24px;
                        border-radius: 12px;
                        margin: 12px 0;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                        transition: transform 0.2s ease-in-out;
                        position: relative;
                    }
                    .advice-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
                    }
                    .advice-title {
                        font-size: 18px;
                        font-weight: 600;
                        margin-bottom: 12px;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    .advice-content {
                        font-size: 16px;
                        line-height: 1.6;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Build 2-column layout for the 4 pieces of advice
            for i in range(0, len(advice_list), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(advice_list):
                        advice = advice_list[i + j]
                        
                        with cols[j]:
                            # Force text color to black in style
                            st.markdown(f"""
                                <div class="advice-card" 
                                     style="background-color: {advice['color']}; color: #000;">
                                    <div class="advice-title" style="color: #000;">
                                        <span class="advice-icon">{advice['icon']}</span>
                                        {advice['title']}
                                    </div>
                                    <div class="advice-content" style="color: #000;">
                                        {advice['advice']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            save_key = f"save_advice_{i+j}"
                            if st.button("💾 Save Advice", key=save_key):
                                if not parent_name:
                                    st.warning("Please log in to save advice")
                                else:
                                    success, error = supabase_manager.save_item_to_supabase(
                                        parent_id=parent_name,
                                        item_type="advice",
                                        title=f"{advice['icon']} {advice['title']}",
                                        content=advice['advice'],
                                        metadata={
                                            "icon": advice['icon'],
                                            "color": advice['color']
                                        }
                                    )
                                    if success:
                                        st.success("Advice saved!")
                                    else:
                                        st.error(f"Failed to save advice: {error}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
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
    """Display conversation starters with save functionality, forcing black text for readability."""
    st.markdown("<h2 class='section-header'>Conversation Starters</h2>", unsafe_allow_html=True)
    
    if not situation:
        st.warning("Please describe the situation to get conversation starters.")
        return

    try:
        messages = [
            {
                "role": "system",
                "content": f"""
                    Generate 5 complete, clear conversation starters for a parent talking to their child about {situation}.
                    Each starter should be a full question or statement.
                    Focus on open-ended, empathetic approaches that encourage dialogue.

                    Format each starter as a JSON array of objects with properties:
                    - text: the complete conversation starter
                    - category: a category from: "Feelings", "Understanding", "Activities", "Solutions", "Ideas"
                    - number: starter number (1-5)
                    - icon: an emoji that matches the category (💭 for Feelings, 🤝 for Understanding, 🎮 for Activities, ⭐ for Solutions, 💡 for Ideas)
                    - color: background color (from ["#EBF4FF", "#FEF3F2", "#F0FDF4", "#FDF2F8", "#FDF6B2"])
                """
            },
            {
                "role": "user",
                "content": f"Generate conversation starters for this situation: {situation}"
            }
        ]
        
        with st.spinner("Generating conversation starters..."):
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            # Parse the JSON array returned
            starters = json.loads(completion.choices[0].message.content)
            
            # CSS for the starter card layout
            st.markdown("""
                <style>
                    .starter-card {
                        border-radius: 12px;
                        padding: 24px;
                        margin-bottom: 20px;
                        border: 1px solid #e2e8f0;
                        transition: transform 0.2s, box-shadow 0.2s;
                        position: relative;
                    }
                    .starter-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .category-badge {
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        padding: 6px 12px;
                        border-radius: 20px;
                        font-size: 0.875rem;
                        font-weight: 500;
                        margin-bottom: 12px;
                        background-color: rgba(255, 255, 255, 0.9);
                        color: #000; /* ensure black text on badge */
                    }
                    .starter-text {
                        font-size: 1.1rem;
                        line-height: 1.6;
                        font-weight: 500;
                        margin-top: 12px;
                    }
                    .starter-number {
                        position: absolute;
                        top: -8px;
                        left: -8px;
                        background-color: #4F46E5;
                        color: white;
                        width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 12px;
                        font-weight: 600;
                        border: 2px solid white;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Display each starter
            for starter in starters:
                # Make the entire card text black for readability
                st.markdown(f"""
                    <div class="starter-card" style="background-color: {starter['color']}; color: #000;">
                        <div class="starter-number">{starter['number']}</div>
                        <div class="category-badge">
                            {starter.get('icon', '💭')} {starter['category']}
                        </div>
                        <div class="starter-text">
                            "{starter['text']}"
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Save button for this starter
                save_key = f"save_starter_{starter['number']}"
                if st.button("💾 Save Starter", key=save_key):
                    if not st.session_state.get('parent_name'):
                        st.warning("Please log in to save conversation starters")
                    else:
                        success, error = supabase_manager.save_item_to_supabase(
                            parent_id=st.session_state['parent_name'],
                            item_type="starter",
                            title=f"{starter['icon']} {starter['category']}",
                            content=starter['text'],
                            metadata={
                                "category": starter['category'],
                                "icon": starter['icon'],
                                "number": starter['number'],
                                "background_color": starter['color']
                            }
                        )
                        if success:
                            st.success("Conversation starter saved!")
                        else:
                            st.error(f"Failed to save: {error}")
            
    except Exception as e:
        print(f"Error generating conversation starters: {e}")
        traceback.print_exc()
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
                    
                    Format as JSON array with objects containing:
                    - title: Strategy name with emoji
                    - purpose: Single clear sentence
                    - steps: Array of 3 action steps
                    - example: Brief specific example
                    - color: Background color (use: ["#7C3AED", "#0D9488", "#D97706"])
                """},
                {"role": "user", "content": f"Generate parenting communication strategies for this situation: {situation}"}
            ]
            
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )

            strategies = json.loads(completion.choices[0].message.content)
            
            st.markdown("""
                <style>
                    .strategy-card {
                        background-color: white;
                        border-radius: 12px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin-bottom: 20px;
                        overflow: hidden;
                        position: relative;
                    }
                    
                    .save-technique-btn {
                        position: absolute;
                        top: 10px;
                        right: 10px;
                        z-index: 10;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            cols = st.columns(3)
            
            for idx, (strategy, col) in enumerate(zip(strategies, cols)):
                with col:
                    # Create unique key for save button
                    save_key = f"save_technique_{idx}"
                    
                    card = f"""
                        <div class="strategy-card">
                            <div class="card-header" style="background-color: {strategy['color']}; color: white; padding: 16px;">
                                {strategy['title']}
                            </div>
                            <div style="padding: 20px;">
                                <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; margin-bottom: 16px;">
                                    {strategy['purpose']}
                                </div>
                                <div class="steps">
                                    {"".join([f'<div style="padding: 12px; margin-bottom: 8px; border: 1px solid #e5e7eb; border-radius: 6px;">{step}</div>' for step in strategy['steps']])}
                                </div>
                                <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; font-style: italic; margin-top: 16px;">
                                    <strong>✨ Example:</strong><br>
                                    {strategy['example']}
                                </div>
                            </div>
                        </div>
                    """
                    
                    st.markdown(card, unsafe_allow_html=True)
                    
                    # Add save button below the card
                    if st.button("💾 Save Technique", key=save_key):
                        if not st.session_state.get('parent_name'):
                            st.warning("Please log in to save techniques")
                        else:
                            # Format content for saving - fixed formatting
                            content = f"""Purpose: {strategy['purpose']}\n\n""" + \
                                    f"""Steps:\n""" + \
                                    "\n".join([f"• {step}" for step in strategy['steps']]) + \
                                    f"""\n\nExample: {strategy['example']}"""
                            
                            success, error = supabase_manager.save_item_to_supabase(
                                parent_id=st.session_state['parent_name'],
                                item_type="technique",
                                title=strategy['title'],
                                content=content,
                                metadata={
                                    "color": strategy['color'],
                                    "situation": situation,
                                    "steps": strategy['steps']
                                }
                            )
                            
                            if success:
                                st.success("Technique saved successfully!")
                            else:
                                st.error(f"Failed to save technique: {error}")
    
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        traceback.print_exc()
        st.error("Unable to generate communication techniques. Please try again.")

def simulate_conversation_streamlit(name: str, child_age: str, situation: str):
    """Display and handle the role-play simulation interface with improved UI"""
    st.markdown("""
        <style>
        .stTextArea textarea {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1rem;
            padding: 0.75rem;
        }
        
        .stTextArea textarea:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            max-width: 80%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .parent-message {
            background-color: #f8fafc;
            margin-left: auto;
            border: 1px solid #e2e8f0;
        }
        
        .child-message {
            background-color: #fff;
            margin-right: auto;
            border: 1px solid #e2e8f0;
        }
        
        .feedback-cloud {
            background-color: #f1f5f9;
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #4b5563;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    # Instructions
    st.markdown("### Getting Started")
    with st.expander("Click for instructions"):
        st.markdown("""
            1. **Customize Your Child's Communication Style**: Describe your child's typical communication for personalized role play.
            2. **Choose Strategy**: Select which communication approach you want to practice.
            3. **Start Conversation**: Begin the role-play and receive real-time feedback.
        
            
            Remember: This is a safe space to practice different approaches and learn from the interaction.
        """)

    # Child's Response Style Section
    st.markdown("### Child's Response Style")
    display_persona_customization()

    # Communication Strategy Section
    st.markdown("### Communication Strategy")
    with st.expander("Select your communication strategy"):
        strategy = st.selectbox(
            "Choose your approach:",
            ["Active Listening", "Positive Reinforcement", "Reflective Questioning"],
            key=f"strategy_select_{st.session_state['simulation_id']}"
        )
        
        if strategy in STRATEGY_EXPLANATIONS:
            st.markdown(STRATEGY_EXPLANATIONS[strategy], unsafe_allow_html=True)
    
    st.session_state['strategy'] = strategy

    # Conversation Section
    st.markdown("### Conversation")
    st.info(f"Current Situation: {situation}")

    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
        
    for msg in st.session_state['conversation_history']:
        speaker = "You" if msg['role'] == 'parent' else st.session_state.get('child_name', 'Child')
        message_class = 'parent-message' if msg['role'] == 'parent' else 'child-message'
        
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"""
                <div class='chat-message {message_class}'>
                    <strong>{speaker}:</strong> {msg['content']}
                </div>
            """, unsafe_allow_html=True)
        
        if msg.get('feedback') and msg['role'] == 'parent':
            with cols[1]:
                st.markdown(f"""
                    <div class='feedback-cloud'>
                        💡 {msg['feedback']['hint']}
                    </div>
                """, unsafe_allow_html=True)

    user_input = st.text_area(
        "Your response",
        placeholder="Type your response here...",
        key=f"parent_input_{st.session_state['simulation_id']}_{st.session_state.get('turn_count', 0)}",
        height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        send_button = st.button("Send Response", type="primary", use_container_width=True)
    with col2:
        end_button = st.button("End Conversation", type="secondary", use_container_width=True)

    if send_button or end_button:
        handle_conversation_input(send_button, end_button, user_input, child_age, situation)

    if st.session_state.get('simulation_ended', False):
        st.success("""
            Simulation completed! You can:
            - Review the conversation above
            - Start a new simulation with different strategies
            - Try different communication approaches
        """)

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
    """Display the initial information collection screen with Supabase integration"""
    st.markdown("<h1 class='main-header'>Welcome to Parenting Support Bot</h1>", unsafe_allow_html=True)
    
    if 'parent_info_id' in st.session_state:
        success, parent_info = supabase_manager.get_parent_information(st.session_state.get('prolific_id'))
        if success and parent_info:
            st.info("Welcome back! We found your previous information.")
            parent_name = parent_info.get('prolific_id', '')
            child_name = parent_info.get('child_name', '')
            child_age = parent_info.get('child_age', '')
            situation = parent_info.get('situation', '')
        else:
            parent_name = child_name = child_age = situation = ''
    else:
        parent_name = child_name = child_age = situation = ''
    
    with st.form(key='parent_info_form'):
        st.markdown("<h2 class='section-header'>Please Tell Us About You</h2>", unsafe_allow_html=True)
        
        st.markdown("""
            <p class='description-text'>
                Please enter your <b>Prolific ID</b> (24-character identifier from your Prolific account)
            </p>
        """, unsafe_allow_html=True)
        
        prolific_id = st.text_input("Prolific ID", 
                                  value=parent_name,
                                  placeholder="Enter your 24-character Prolific ID...",
                                  help="This is the ID assigned to you by Prolific")
        
        child_name = st.text_input("Child's Name or Nickname", 
                                 value=child_name,
                                 placeholder="Enter your child's name or pseudonym...")
        
        child_age = st.selectbox("Child's Age Range",
                               ["3-5 years", "6-9 years", "10-12 years"],
                               index=["3-5 years", "6-9 years", "10-12 years"].index(child_age) if child_age else 0)
        
        situation = st.text_area("Situation Description",
                               value=situation,
                               placeholder="Type your situation here...",
                               height=120)
        
        submit_button = st.form_submit_button("Start", use_container_width=True)
        
        if submit_button:
            if not prolific_id or not child_name or not situation:
                st.error("Please fill in all fields")
            elif len(prolific_id) != 24:
                st.error("Please enter a valid 24-character Prolific ID")
            else:
                parent_data = {
                    'prolific_id': prolific_id,
                    'child_name': child_name,
                    'child_age': child_age,
                    'situation': situation,
                }
                
                success, result = supabase_manager.save_parent_information(parent_data)
                
                if success:
                    st.session_state['parent_info_id'] = result
                    st.session_state['parent_name'] = prolific_id
                    st.session_state['child_name'] = child_name
                    st.session_state['child_age'] = child_age
                    st.session_state['situation'] = situation
                    st.session_state['info_submitted'] = True
                    st.rerun()
                else:
                    st.error(f"Failed to save information: {result}")

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
    """Reset simulation-specific session state variables"""
    st.session_state['conversation_history'] = []
    st.session_state['turn_count'] = 0
    st.session_state['simulation_id'] = str(uuid4())
    st.session_state['simulation_ended'] = False
    st.session_state['stored_responses'] = {}
    
    if all(st.session_state.get(k) for k in ['child_persona', 'situation', 'child_age']):
        try:
            st.session_state['child_mood'] = determine_child_mood(
                st.session_state['child_persona'],
                st.session_state['situation'],
                st.session_state['child_age']
            )
        except Exception as e:
            print(f"Error resetting simulation mood: {e}")
            st.error("Unable to determine child's response style for new simulation.")

def analyze_child_persona(persona: dict) -> dict:
    """
    Analyzes persona to determine mood tendencies based on communication patterns,
    emotional expression, and special traits.
    """
    weights = {'cooperative': 1.0, 'defiant': 1.0, 'distracted': 1.0}
    
    if not isinstance(persona, dict):
        raise ValueError("Invalid persona format")

    communication_patterns = {
        'cooperative': [
            'listens well', 'understands', 'calm', 'helps', 'agrees', 'follows',
            'shares', 'patient', 'works together', 'polite', 'kind'
        ],
        'defiant': [
            'argues', 'refuses', 'angry', 'no', 'won\'t', 'never', 'protests',
            'yells', 'fights', 'resists', 'ignores', 'oppositional'
        ],
        'distracted': [
            'changes subject', 'unfocused', 'busy', 'forgets', 'wanders off',
            'looks away', 'fidgets', 'moves around', 'daydreams'
        ]
    }

    style_text = ' '.join([
        str(persona.get('communication_style', '')),
        str(persona.get('typical_phrases', '')),
        str(persona.get('response_to_no', ''))
    ]).lower()

    for mood, patterns in communication_patterns.items():
        matches = sum(1 for pattern in patterns if pattern in style_text)
        weights[mood] += matches * 0.3

    behavior_weights = {
        'cooperative': [
            'Shows physical affection',
            'Takes time to process',
            'Uses humor positively'
        ],
        'defiant': [
            'Argues and debates frequently',
            'Gets loud when excited',
            'Becomes defensive easily'
        ],
        'distracted': [
            'Changes subject often',
            'Gives minimal responses',
            'Physically expressive'
        ]
    }

    patterns = persona.get('response_patterns', [])
    for mood, behaviors in behavior_weights.items():
        matches = sum(1 for behavior in behaviors if behavior in patterns)
        weights[mood] += matches * 0.4

    expression_style = persona.get('emotion_style')
    if expression_style:
        expression_adjustments = {
            'Very Reserved': {'cooperative': 0.4, 'defiant': -0.2, 'distracted': 0.1},
            'Somewhat Reserved': {'cooperative': 0.2, 'defiant': -0.1},
            'Balanced': {},
            'Somewhat Expressive': {'defiant': 0.2, 'distracted': 0.2},
            'Very Expressive': {'defiant': 0.4, 'distracted': 0.3, 'cooperative': -0.2}
        }
        
        if expression_style in expression_adjustments:
            for mood, adjustment in expression_adjustments[expression_style].items():
                weights[mood] += adjustment

    special_traits = persona.get('special_traits', [])
    trait_adjustments = {
        'Sensitive to tone of voice': {'defiant': 0.3, 'cooperative': -0.1},
        'Needs time to process changes': {'distracted': 0.2, 'defiant': 0.1},
        'Gets overwhelmed easily': {'distracted': 0.3, 'defiant': 0.2},
        'Strong opinions': {'defiant': 0.3},
        'Literal interpretation': {'cooperative': 0.2}
    }

    for trait in special_traits:
        if trait in trait_adjustments:
            for mood, adjustment in trait_adjustments[trait].items():
                weights[mood] += adjustment

    return weights

def analyze_situation_context(situation: str, child_age: str) -> dict:
    situation_lower = situation.lower()
    mood_weights = {'cooperative': 1.0, 'defiant': 1.0, 'distracted': 1.0}

    situation_factors = {
        'cooperative': {
            'positive_activities': [
                'play', 'game', 'park', 'fun', 'together', 'help', 'reward',
                'special time', 'reading', 'movie', 'craft', 'drawing',
                'outdoor', 'sports', 'swimming', 'bike'
            ],
            'learning_moments': [
                'new skill', 'learning', 'practice', 'trying', 'improvement',
                'achievement', 'success', 'understand', 'explore', 'discover'
            ],
            'social_situations': [
                'friends', 'family', 'sharing', 'cooperation', 'team',
                'playing together', 'helping others', 'birthday', 'celebration'
            ],
            'positive_routines': [
                'morning routine', 'getting ready', 'preparing', 'organizing',
                'planning', 'choosing', 'deciding'
            ]
        },
        'defiant': {
            'challenging_routines': [
                'bedtime', 'sleep', 'nap', 'wake up', 'morning',
                'homework', 'study', 'school work', 'assignment',
                'chores', 'clean', 'tidy', 'organize', 'put away'
            ],
            'limit_setting': [
                'stop', 'no', 'don\'t', 'cannot', 'not allowed', 'limit',
                'restrict', 'time\'s up', 'finish', 'end', 'later',
                'screen time', 'device', 'tablet', 'phone', 'tv'
            ],
            'transitions': [
                'change activity', 'switch', 'move to', 'time to go',
                'leave', 'get ready', 'hurry', 'quick', 'rush'
            ],
            'frustrating_situations': [
                'difficult', 'challenge', 'problem', 'mistake', 'wrong',
                'unfair', 'not working', 'broken', 'lost', 'waiting'
            ]
        },
        'distracted': {
            'competing_activities': [
                'playing', 'watching', 'game', 'tv', 'video',
                'friends', 'toys', 'device', 'phone', 'tablet',
                'computer', 'screen', 'favorite show'
            ],
            'physical_states': [
                'tired', 'hungry', 'sleepy', 'exhausted', 'restless',
                'energetic', 'excited', 'hyper', 'fidgety'
            ],
            'environmental_factors': [
                'noise', 'busy', 'crowd', 'new place', 'unfamiliar',
                'interesting', 'exciting', 'loud', 'movement'
            ],
            'timing_factors': [
                'late', 'rush', 'hurry', 'busy day', 'long day',
                'after school', 'before dinner', 'evening'
            ]
        }
    }

    for mood, categories in situation_factors.items():
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in situation_lower:
                    mood_weights[mood] += 0.2 if len(keyword.split()) == 1 else 0.3

    age_factors = {
        "3-5 years": {
            'distracted': 1.3,
            'defiant': 1.2,
            'keywords': ['nap', 'play', 'share', 'toy', 'snack']
        },
        "6-9 years": {
            'cooperative': 1.1,
            'keywords': ['school', 'homework', 'friends', 'games', 'activities']
        },
        "10-12 years": {
            'defiant': 1.1,
            'keywords': ['independence', 'social', 'screen time', 'responsibility']
        }
    }

    if child_age in age_factors:
        age_info = age_factors[child_age]
        for mood, adjustment in age_info.items():
            if mood in mood_weights and isinstance(adjustment, (float, int)):
                mood_weights[mood] *= adjustment
        for keyword in age_info.get('keywords', []):
            if keyword in situation_lower:
                mood_weights['cooperative'] += 0.2

    emotional_indicators = {
        'cooperative': ['happy', 'excited', 'eager', 'interested', 'calm', 'relaxed'],
        'defiant': ['angry', 'upset', 'frustrated', 'mad', 'annoyed', 'refusing'],
        'distracted': ['overwhelmed', 'tired', 'excited', 'anxious', 'nervous']
    }

    for mood, indicators in emotional_indicators.items():
        for indicator in indicators:
            if indicator in situation_lower:
                mood_weights[mood] += 0.25

    return mood_weights

def update_child_mood(persona: dict):
    """Updates child's mood based on persona and situation with improved error handling"""
    if not persona:
        print("No persona provided for mood update")
        return False
        
    try:
        situation = st.session_state.get('situation', '')
        child_age = st.session_state.get('child_age', '')
        
        if not situation or not child_age:
            print("Missing required context for mood update")
            return False
        
        new_mood = determine_child_mood(persona, situation, child_age)
        
        if new_mood:
            st.session_state['child_mood'] = new_mood
            print(f"Updated mood to: {new_mood}")
            return True
        else:
            print("Mood determination returned None")
            return False
            
    except Exception as e:
        print(f"Error updating child mood: {str(e)}")
        traceback.print_exc()
        return False

def display_saved_items():
    """Display saved items in an organized, filterable view"""
    st.markdown("<h2 class='section-header'>Your Saved Content</h2>", unsafe_allow_html=True)
    
    if not st.session_state.get('parent_name'):
        st.warning("Please log in to view saved content")
        return
        
    # Add filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        filter_type = st.multiselect(
            "Filter by type:",
            options=["advice", "technique", "starter"],
            default=["advice", "technique", "starter"]
        )
    
    success, items = supabase_manager.get_saved_items(st.session_state['parent_name'])
    
    if not success:
        st.error("Failed to retrieve saved items")
        return
        
    if not items:
        st.info("You haven't saved any content yet. Browse through the different sections and click 'Save' on items you want to keep!")
        return
        
    # Filter items based on selection
    filtered_items = [item for item in items if item['item_type'] in filter_type]
    
    st.markdown("""
        <style>
        .saved-item {
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            position: relative;
            transition: transform 0.2s ease-in-out;
        }
        
        .saved-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .item-type-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            background-color: rgba(255,255,255,0.2);
        }
        
        .item-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .item-content {
            font-size: 14px;
            line-height: 1.6;
            margin-top: 12px;
            margin-bottom: 16px;
        }
        
        .item-metadata {
            font-size: 12px;
            border-top: 1px solid rgba(255,255,255,0.2);
            padding-top: 12px;
            margin-top: 12px;
        }

        .item-icon {
            font-size: 24px;
            margin-right: 8px;
        }

        .delete-button {
            position: absolute;
            bottom: 10px;
            right: 10px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            background-color: rgba(255,255,255,0.2);
            color: white;
            border: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .delete-button:hover {
            background-color: rgba(255,255,255,0.3);
        }
        </style>
    """, unsafe_allow_html=True)
    
    for item in filtered_items:
        item_type = item['item_type']
        metadata = item.get('metadata', {})
        
        # Get background color from metadata or use default colors
        type_colors = {
            'advice': '#4338CA',    # Deep blue
            'technique': '#047857',  # Deep green
            'starter': '#BE185D'     # Deep pink
        }
        
        bg_color = metadata.get('background_color') or type_colors.get(item_type, '#4B5563')
        icon = metadata.get('icon', '📝')
        
        st.markdown(f"""
            <div class="saved-item" style="background-color: {bg_color};">
                <div class="item-type-badge" style="color: white;">
                    {item_type.title()}
                </div>
                <div class="item-title" style="color: white;">
                    {item['title']}
                </div>
                <div class="item-content" style="color: white;">
                    {item['content']}
                </div>
                <div class="item-metadata" style="color: rgba(255,255,255,0.8);">
                    Saved on: {datetime.fromisoformat(item['saved_at']).strftime('%B %d, %Y %I:%M %p')}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Delete", key=f"delete_{item['id']}", type="secondary"):
                success, error = supabase_manager.delete_saved_item(item['id'])
                if success:
                    st.toast("Item deleted!")
                    time.sleep(0.1)  # Very short delay
                    st.rerun()
                else:
                    st.error(f"Failed to delete item: {error}")

def main():
    def test_supabase_connection():
        """Test Supabase connection and configuration"""
        print("\n=== Testing Supabase Connection ===")
        
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
        
        try:
            manager = SupabaseManager()
            success = manager.initialize()
            if success:
                print("✅ Supabase connection successful")
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

    if not test_supabase_connection():
        st.error("Failed to connect to database. Please check configuration.")
        st.stop()

    feature_order = {
        "Advice": "Get expert guidance on handling specific parenting situations based on evidence-based strategies.",
        "Communication Techniques": "Discover helpful ways to communicate with your child and get tips on how to address your specific situation.",
        "Conversation Starters": "Receive help initiating difficult conversations with suggested opening phrases and questions.",
        "Role-Play Simulation": "Practice conversations in a safe environment to develop and refine your communication approach.",
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
        
        # Add a divider
        st.markdown("---")
        
        # Add Saved Items button
        if st.button("📚 View Saved Items", key="view_saved_items", use_container_width=True):
            st.session_state['show_saved_items'] = True
            st.rerun()

        # Display progress sidebar
        st.markdown("---")
        display_progress_sidebar(feature_order)

    if 'show_tutorial' not in st.session_state:
        show_tutorial()
    else:
        st.markdown("<h1 class='main-header'>Parenting Support Bot</h1>", unsafe_allow_html=True)

        # Show either saved items or main content
        if st.session_state.get('show_saved_items', False):
            display_saved_items()
            if st.button("← Back to Main Menu", type="secondary"):
                st.session_state['show_saved_items'] = False
                st.rerun()
        else:
            selected = st.radio(
                "Choose an option:",
                list(feature_order.keys()),
                horizontal=True,
                help="Select a tool that best matches your current needs"
            )

            st.info(feature_order[selected])

            if selected == "Advice":
                track_feature_visit("advice")
                display_advice(
                    st.session_state['parent_name'],
                    st.session_state['child_age'],
                    st.session_state['situation']
                )
            elif selected == "Communication Techniques":
                track_feature_visit("communication_techniques")
                display_communication_techniques(st.session_state['situation'])
            elif selected == "Conversation Starters":
                track_feature_visit("conversation_starters")
                display_conversation_starters(st.session_state['situation'])
            elif selected == "Role-Play Simulation":
                track_feature_visit("role_play")
                simulate_conversation_streamlit(
                    st.session_state['parent_name'],
                    st.session_state['child_age'],
                    st.session_state['situation']
                )
            elif selected == "Saved Items":
                track_feature_visit("saved_items")
                display_saved_items()

    # Update session analytics
    if 'session_start_time' in st.session_state:
        current_time = datetime.now()
        st.session_state['session_duration'] = (current_time - st.session_state['session_start_time']).total_seconds()
        
        # Update last interaction time
        st.session_state['last_interaction_time'] = current_time
        
    # Check for session timeout (30 minutes)
    if 'last_interaction_time' in st.session_state:
        time_since_last_interaction = (datetime.now() - st.session_state['last_interaction_time']).total_seconds()
        if time_since_last_interaction > 1800:  # 30 minutes
            st.warning("Your session has been inactive for 30 minutes. Please refresh the page to continue.")

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
        traceback.print_exc()