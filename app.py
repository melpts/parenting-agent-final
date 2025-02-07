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

# Data processing imports
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

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

# Citation imports
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


COMPONENT_CSS = """
<style>
/* Progress Item Styles */
.progress-item {
    display: flex;
    align-items: center;
    padding: 8px;
    margin: 4px 0;
    border-radius: 6px;
    transition: background-color 0.2s;
}

.progress-item:hover {
    background-color: #F3F4F6;
}

.progress-item.completed {
    color: #059669;
}

.progress-icon {
    margin-right: 8px;
}

/* Review Styles */
.review-summary {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}

.review-summary h4 {
    color: #2563EB;
    margin-bottom: 12px;
}

.review-summary ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.review-summary li {
    padding: 4px 0;
    color: #4B5563;
}

.detailed-analysis {
    background: #F9FAFB;
    padding: 16px;
    border-radius: 8px;
    margin-top: 12px;
}

.detailed-analysis h4 {
    color: #1F2937;
    margin-bottom: 12px;
}

.detailed-analysis ul {
    list-style: none;
    padding: 0;
}

.detailed-analysis li {
    padding: 4px 0;
    color: #4B5563;
}

/* Form Description Styles */
.form-description {
    color: #6B7280;
    font-size: 0.875rem;
    margin-bottom: 16px;
    padding: 12px;
    background: #F3F4F6;
    border-radius: 6px;
}

/* Saved Items Styles */
.saved-item {
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
    color: white;
    position: relative;
}

.item-header {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
}

.item-icon {
    font-size: 1.5rem;
    margin-right: 12px;
}

.item-title {
    font-size: 1.125rem;
    font-weight: 500;
}

.item-type {
    position: absolute;
    top: 12px;
    right: 12px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    background: rgba(255, 255, 255, 0.2);
}

.item-content {
    margin: 12px 0;
    line-height: 1.6;
}

.item-metadata {
    font-size: 0.75rem;
    opacity: 0.8;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
}

/* Active Persona Banner */
.active-persona-banner {
    background: #F0F9FF;
    border: 1px solid #BAE6FD;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
}

.persona-info {
    display: flex;
    align-items: center;
    gap: 8px;
}

.persona-label {
    color: #0369A1;
    font-weight: 500;
}

.persona-name {
    color: #0C4A6E;
    font-weight: 600;
}

.persona-style {
    color: #0369A1;
    font-size: 0.875rem;
}

/* Responsive Adjustments */
@media (max-width: 768px) {
    .saved-item {
        margin: 8px 0;
    }
    
    .item-type {
        position: static;
        display: inline-block;
        margin-top: 8px;
    }
    
    .detailed-analysis {
        padding: 12px;
    }
}
</style>
"""

# Apply component styles
st.markdown(COMPONENT_CSS, unsafe_allow_html=True)

# Enhanced CSS with new components
ENHANCED_CSS = """
<style>
/* Base Typography and Layout */
body {
    font-family: 'Inter', sans-serif;
    color: #1a1a1a;
    line-height: 1.6;
}

/* Enhanced Headers */
.main-header {
    font-size: 2.5em;
    font-weight: 800;
    color: #2563eb;
    margin-bottom: 1.5em;
}

.section-header {
    font-size: 2em;
    font-weight: 700;
    color: #2563eb;
    margin: 1.2em 0;
    border-bottom: 3px solid #60a5fa;
    padding-bottom: 0.5em;
}

/* Wizard Steps Component */
.wizard-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.step-indicator {
    margin-bottom: 2rem;
}

.step-progress {
    height: 4px;
    background: #e5e7eb;
    border-radius: 2px;
    margin: 1rem 0;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    background: #4338CA;
    transition: width 0.3s ease;
}

.step-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
    color: #6b7280;
    font-size: 0.875rem;
}

/* Enhanced Chat Interface */
.chat-container {
    max-width: 800px;
    margin: 20px auto;
    padding: 20px;
}

.message-bubble {
    max-width: 80%;
    padding: 16px;
    margin: 8px 0;
    border-radius: 16px;
    position: relative;
    line-height: 1.5;
    transition: all 0.2s ease-in-out;
}

.message-bubble:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.parent-message {
    background-color: #4338CA;
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}

.child-message {
    background-color: #F3F4F6;
    color: #1F2937;
    margin-right: auto;
    border-bottom-left-radius: 4px;
}

/* Enhanced Feedback Display */
.feedback-bubble {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 12px;
    margin-top: 8px;
    border-radius: 8px;
    font-size: 0.875rem;
}

.feedback-positive {
    color: #059669;
    background-color: #ECFDF5;
    padding: 8px;
    border-radius: 6px;
    margin-bottom: 4px;
}

.feedback-suggestion {
    color: #D97706;
    background-color: #FFFBEB;
    padding: 8px;
    border-radius: 6px;
}

/* Enhanced Review Section */
.review-container {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.review-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e5e7eb;
}

.review-content {
    margin-top: 12px;
}

.review-details {
    background: #f9fafb;
    padding: 12px;
    border-radius: 8px;
    margin-top: 8px;
}

/* Enhanced Form Elements */
.form-section {
    background: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Enhanced Buttons */
.action-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-weight: 500;
    transition: all 0.2s;
}

.primary-button {
    background-color: #4338CA;
    color: white;
}

.primary-button:hover {
    background-color: #4F46E5;
    transform: translateY(-1px);
}

.secondary-button {
    background-color: #F3F4F6;
    color: #374151;
}

.secondary-button:hover {
    background-color: #E5E7EB;
    transform: translateY(-1px);
}

/* Enhanced Persona Card */
.persona-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: 1px solid #e5e7eb;
}

.persona-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.persona-header {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
}

.persona-icon {
    font-size: 1.5rem;
    margin-right: 12px;
}

.persona-name {
    font-weight: 600;
    color: #111827;
}

.persona-details {
    color: #6B7280;
    font-size: 0.875rem;
}

/* Behavior Tags */
.behavior-tag {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    background: #F3F4F6;
    border-radius: 9999px;
    font-size: 0.875rem;
    color: #374151;
    margin: 4px;
    transition: all 0.2s;
}

.behavior-tag.selected {
    background: #4338CA;
    color: white;
}

.behavior-tag:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Enhanced Inputs */
.enhanced-input {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #e5e7eb;
    border-radius: 0.375rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.enhanced-input:focus {
    border-color: #4338CA;
    box-shadow: 0 0 0 3px rgba(67, 56, 202, 0.1);
    outline: none;
}

/* Responsive Design */
@media (max-width: 768px) {
    .wizard-container {
        padding: 1rem;
    }
    
    .message-bubble {
        max-width: 90%;
    }
    
    .behavior-tag {
        width: 100%;
        justify-content: center;
        margin: 4px 0;
    }
}

/* Animation Classes */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-slide-in {
    animation: slideIn 0.3s ease-out forwards;
}

/* Accessibility Enhancements */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
}

/* Focus Styles */
:focus {
    outline: 2px solid #4338CA;
    outline-offset: 2px;
}

/* Dark Mode Support */
@media (prefers-color-scheme: dark) {
    .persona-card {
        background: #1F2937;
        border-color: #374151;
    }
    
    .persona-name {
        color: #F3F4F6;
    }
    
    .persona-details {
        color: #9CA3AF;
    }
}

.message-wrapper {
    display: flex;
    margin-bottom: 10px;
}
.message-right {
    justify-content: flex-end;
}
.message-left {
    justify-content: flex-start;
}
.feedback-bubble {
    background-color: #ECFDF5;
    border: 1px solid #059669;
    padding: 8px;
    margin-top: 8px;
    border-radius: 6px;
    font-size: 0.9em;
    color: #065F46;
}
.message-wrapper {
    display: flex;
    margin-bottom: 10px;
}
.message-right {
    justify-content: flex-end;
}
.message-left {
    justify-content: flex-start;
}
</style>
"""

# Apply enhanced CSS
st.markdown(ENHANCED_CSS, unsafe_allow_html=True)

# Additional CSS for enhanced components
ADDITIONAL_CSS = """
<style>
/* Advice Card Styles */
.advice-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: transform 0.2s ease-in-out;
    position: relative;
}

.advice-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.advice-category-badge {
    position: absolute;
    top: 12px;
    right: 12px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background-color: rgba(255,255,255,0.2);
}

.advice-title {
    font-size: 18px;
    font-weight: 600;
    margin: 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.advice-content {
    font-size: 16px;
    line-height: 1.6;
    margin: 16px 0;
}

.advice-actions {
    margin-top: 16px;
    display: flex;
    justify-content: flex-end;
}

/* Communication Techniques Styles */
.strategy-card {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    transition: transform 0.2s ease;
}

.strategy-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.strategy-header {
    color: white;
    padding: 16px;
}

.strategy-header h3 {
    margin: 0;
    font-size: 1.2em;
}

.strategy-content {
    padding: 20px;
}

.purpose-section {
    background-color: #f8fafc;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 16px;
}

.steps-section {
    margin: 16px 0;
}

.step-item {
    padding: 8px;
    margin: 8px 0;
    background: #f3f4f6;
    border-radius: 6px;
}

.example-section, .outcome-section {
    background-color: #f8fafc;
    padding: 12px;
    border-radius: 8px;
    margin-top: 16px;
}

/* Conversation Starters Styles */
.starter-card {
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.2s ease;
}

.starter-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.starter-category {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
    background-color: rgba(255,255,255,0.9);
    margin-bottom: 12px;
}

.starter-text {
    font-size: 1.1rem;
    line-height: 1.6;
    font-weight: 500;
    margin: 16px 0;
}

.starter-approach {
    font-size: 0.9rem;
    background-color: rgba(255,255,255,0.7);
    padding: 12px;
    border-radius: 8px;
    margin-top: 12px;
}

/* Save Button Styles */
.save-button {
    background-color: rgba(255,255,255,0.9);
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}

.save-button:hover {
    background-color: white;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Enhanced Helper Text */
.helper-text {
    color: #6B7280;
    font-size: 0.875rem;
    margin: 4px 0 12px 0;
}

/* Enhanced Feedback Styles */
.feedback-message {
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.feedback-success {
    background-color: #ECFDF5;
    color: #065F46;
}

.feedback-warning {
    background-color: #FFFBEB;
    color: #92400E;
}

.feedback-error {
    background-color: #FEF2F2;
    color: #991B1B;
}

/* Loading Spinner Enhancement */
.loading-spinner {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
}

/* Responsive Adjustments */
@media (max-width: 768px) {
    .advice-card, .strategy-card, .starter-card {
        margin: 12px 0;
    }
    
    .advice-category-badge {
        position: static;
        margin-bottom: 12px;
    }
    
    .strategy-content {
        padding: 16px;
    }
}

/* Print Styles */
@media print {
    .advice-card, .strategy-card, .starter-card {
        break-inside: avoid;
        page-break-inside: avoid;
    }
}

<style>
/* Enhanced Message Styles */
.message-wrapper {
    display: flex;
    margin-bottom: 10px;
    padding: 0 10px;
}

.message-right {
    justify-content: flex-end;
}

.message-left {
    justify-content: flex-start;
}

.message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.parent-message {
    background-color: #4338CA;
    color: white;
}

.child-message {
    background-color: #F3F4F6;
    color: #1F2937;
}

/* Separate Feedback Bubble */
.feedback-bubble {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 10px 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.feedback-positive {
    background-color: #ECFDF5;
    color: #065F46;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 8px;
}

.feedback-suggestion {
    background-color: #FEF3C7;
    color: #92400E;
    padding: 12px;
    border-radius: 6px;
}

/* Feedback Detail in Review */
.feedback-detail {
    background: #FFFFFF;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
</style>
"""

# Apply additional CSS
st.markdown(ADDITIONAL_CSS, unsafe_allow_html=True)

TUTORIAL_CSS = """
<style>
.tutorial-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 24px;
}

.tutorial-intro {
    font-size: 1.2em;
    color: #374151;
    margin-bottom: 24px;
    text-align: center;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 24px;
    margin: 32px 0;
}

.feature-card {
    background: white;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.2s ease;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.feature-icon {
    font-size: 2.5em;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 1.2em;
    font-weight: 600;
    color: #2563eb;
    margin-bottom: 8px;
}

.feature-desc {
    color: #6B7280;
    font-size: 0.9em;
    line-height: 1.5;
}

@media (max-width: 768px) {
    .feature-grid {
        grid-template-columns: 1fr;
    }
    
    .feature-card {
        padding: 16px;
    }
}
</style>
"""

# Apply tutorial CSS
st.markdown(TUTORIAL_CSS, unsafe_allow_html=True)

CHAT_CSS = """
<style>
.message-wrapper {
    display: flex;
    margin: 8px 0;
    padding: 0 12px;
}

.message {
    max-width: 80%;
    padding: 8px 12px;
    border-radius: 15px;
}

.parent-message {
    background-color: #4338CA;
    color: white;
    margin-left: auto;
}

.child-message {
    background-color: #F3F4F6;
    color: #1F2937;
}

.feedback-bubble {
    margin: 4px 12px;
    padding: 8px 12px;
    border-radius: 4px;
    background-color: #ECFDF5;
    color: #065F46;
    font-size: 0.9em;
    max-width: 400px;
    margin-left: auto;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.feedback-hint {
    font-weight: 500;
}
</style>
"""

# Apply chat CSS
st.markdown(CHAT_CSS, unsafe_allow_html=True)



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
        
    def delete_saved_item(self, item_id: str) -> Tuple[bool, str]:
        """Delete a saved item from the database"""
        if not self.ensure_initialized():
            return False, "Database not initialized"
            
        try:
            result = self.supabase.table('saved_items')\
                .delete()\
                .eq('id', item_id)\
                .execute()
                
            if result.data:
                return True, "Item deleted successfully"
            return False, "Failed to delete item"
            
        except Exception as e:
            error_msg = f"Error deleting item: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg

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
            # Test tables - create if they don't exist
            self._ensure_tables_exist()
            
            self._initialized = True
            print("Supabase initialization successful")
            return True
            
        except Exception as e:
            print(f"Supabase initialization error: {str(e)}")
            traceback.print_exc()
            return False


    
    def _ensure_tables_exist(self):
        try:
            # self.supabase.rpc('exec_sql', {'query': PERSONA_TABLE_SQL}).execute()

            print("✓ Database tables verified")
        except Exception as e:
            print(f"Error ensuring tables exist: {e}")
            raise

    def ensure_initialized(self) -> bool:
        """Helper method to ensure connection is initialized"""
        if not self._initialized and not self.initialize():
            print("Failed to initialize Supabase connection")
            return False
        return True

    def get_parent_information(self, prolific_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Retrieve parent information from Supabase"""
        if not self.ensure_initialized():
            return False, None
    
        try:
            result = self.supabase.table('parent_information')\
                .select("*")\
                .eq('prolific_id', prolific_id)\
                .limit(1)\
                .execute()
                
            if result.data:
                return True, result.data[0]
            return True, None
            
        except Exception as e:
            print(f"Error retrieving parent information: {e}")
            traceback.print_exc()
            return False, None

    def save_parent_information(self, parent_data: dict) -> Tuple[bool, Optional[str]]:
        """Save parent information to Supabase"""
        if not self.ensure_initialized():
            return False, "Failed to initialize Supabase connection"

        try:
            required_fields = ['prolific_id', 'child_name', 'child_age', 'situation']
            missing_fields = [f for f in required_fields if f not in parent_data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            parent_data['created_at'] = datetime.utcnow().isoformat()
            result = self.supabase.table('parent_information').insert(parent_data).execute()
            
            if not result.data:
                raise ValueError("No data returned from insert operation")

            return True, result.data[0].get('id')

        except Exception as e:
            error_msg = f"Error saving parent information: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg

    def save_simulation_data(self, simulation_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not self.ensure_initialized():
            return False, "Failed to initialize Supabase connection"
        try:
            # now require the following keys:
            required_fields = [
                'user_id', 'strategy', 'child_age', 'situation',
                'turn_count', 'parent_message', 'created_at'
            ]
            missing_fields = [f for f in required_fields if f not in simulation_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Build the simulation_data object that you want to store.
            new_simulation_data = {
                "strategy": simulation_data["strategy"],
                "child_age": simulation_data["child_age"],
                "situation": simulation_data["situation"],
                "turn_count": simulation_data["turn_count"],
                "parent_message": simulation_data["parent_message"]
            }
        
            data = {
                "user_id": simulation_data["user_id"],
                "simulation_data": new_simulation_data,
                "created_at": simulation_data["created_at"],
                "completed_at": None,
                "langsmith_run_id": simulation_data.get("langsmith_run_id")
            }
        
            result = self.supabase.table('simulations').insert(data).execute()
            return True, result.data[0].get('id') if result.data else None
        except Exception as e:
            print(f"Error saving simulation: {str(e)}")
            traceback.print_exc()
            return False, None


    def complete_simulation(self, simulation_id: str, run_id: Optional[str] = None) -> Tuple[bool, str]:
        """Mark a simulation as completed"""
        if not self.ensure_initialized():
            return False, "Database not initialized"
            
        try:
            data = {
                "completed_at": datetime.utcnow().isoformat(),
                "langsmith_run_id": run_id
            }
            
            result = self.supabase.table('simulations')\
                .update(data)\
                .eq('id', simulation_id)\
                .execute()
                
            return True, "Simulation completed" if result.data else False, "Failed to complete simulation"
            
        except Exception as e:
            return False, f"Error completing simulation: {str(e)}"

    def save_simulation_analytics(self, simulation_id: str, messages: list) -> bool:
            """
            Calculate detailed simulation analytics from conversation messages and save them
            to the simulation_analytics table. Detailed metrics include:
            - total_exchanges: Number of parent messages.
            - average_response_time: Average response time (in seconds).
            - most_used_strategy: The most frequently used strategy.
            - strategy_usage: Counts per strategy (or a friendly message if none).
            - effective_approaches: Count of messages with a positive feedback hint.
            - growth_areas: Count of messages with detailed feedback suggestions.
            - conversation_timeline: A list of chat messages with role, content, and timestamp.
            
            Also, the parent's Prolific ID is saved in the user_id field.
            """
            if not self.ensure_initialized():
                return False
            try:
                total_exchanges = 0
                strategy_usage = {}
                response_times = []
                effective_approaches = 0
                growth_areas = 0
                conversation_timeline = []
                
                prev_time = None
                for msg in messages:
                    # Record each message for the timeline.
                    conversation_timeline.append({
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "timestamp": msg.get("timestamp")
                    })
                    if msg.get("role") == "parent":
                        total_exchanges += 1
                        # Count strategy usage.
                        if "strategy_used" in msg:
                            strat = msg["strategy_used"]
                            strategy_usage[strat] = strategy_usage.get(strat, 0) + 1
                        # Calculate response time.
                        if "timestamp" in msg:
                            current_time = datetime.fromisoformat(msg["timestamp"])
                            if prev_time is not None:
                                diff = (current_time - prev_time).total_seconds()
                                response_times.append(diff)
                            prev_time = current_time
                        # Count feedback items.
                        if "feedback" in msg:
                            feedback = msg["feedback"]
                            if feedback.get("hint"):
                                effective_approaches += 1
                            if feedback.get("detailed"):
                                growth_areas += 1
                
                avg_response_time = (sum(response_times) / len(response_times)) if response_times else 0.0
                most_used_strategy = "None"
                if strategy_usage:
                    most_used_strategy = max(strategy_usage.items(), key=lambda x: x[1])[0]
                
                # Use a friendly message if there is no strategy usage data.
                strategy_usage_display = strategy_usage if strategy_usage else "No strategy usage data available"
                
                detailed_metrics = {
                    "total_exchanges": total_exchanges,
                    "average_response_time": avg_response_time,
                    "most_used_strategy": most_used_strategy,
                    "strategy_usage": strategy_usage_display,
                    "effective_approaches": effective_approaches,
                    "growth_areas": growth_areas,
                    "conversation_timeline": conversation_timeline
                }
                
                # Get the parent's prolific ID from session_state.
                parent_id = st.session_state.get('parent_name')
                if not parent_id:
                    print("No parent_name found in session_state. Cannot save user_id in analytics.")
                    return False
                
                data = {
                    "simulation_id": simulation_id,
                    "user_id": parent_id,
                    "metrics": detailed_metrics,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                result = self.supabase.table('simulation_analytics').insert(data).execute()
                return bool(result.data)
            except Exception as e:
                print(f"Error saving detailed simulation analytics: {e}")
                traceback.print_exc()
                return False

    def save_item_to_supabase(self, parent_id: str, item_type: str, title: str, content: str, metadata: dict) -> Tuple[bool, str]:
        """Save generic content item to Supabase"""
        if not self.ensure_initialized():
            return False, "Database not initialized"
            
        try:
            data = {
                'parent_id': parent_id,
                'item_type': item_type,
                'title': title,
                'content': content,
                'metadata': metadata,
                'saved_at': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table('saved_items').insert(data).execute()
            
            return (True, "Item saved successfully") if result.data else (False, "Failed to save item")
            
        except Exception as e:
            return False, f"Error saving item: {str(e)}"

class PersonaManager:
    def __init__(self, supabase_manager: SupabaseManager):
        self.supabase = supabase_manager
    
    def load_personas(self, parent_id: str) -> Tuple[bool, List[dict]]:
        """Load all personas for a parent"""
        try:
            result = self.supabase.supabase.table('child_personas')\
                .select("*")\
                .eq('parent_id', parent_id)\
                .order('created_at', desc=True)\
                .execute()
                
            return (True, result.data) if result.data else (True, [])
            
        except Exception as e:
            print(f"Error loading personas: {e}")
            traceback.print_exc()
            return False, []
        
    def save_persona(self, parent_id: str, persona_data: dict) -> Tuple[bool, str]:
        """Save persona to the child_personas table"""
        try:
            if not self.supabase.ensure_initialized():
                return False, "Database not initialized"

            # Validate persona data
            success, msg = self.validate_persona_data(persona_data)
            if not success:
                return False, msg
                
            data = {
                'parent_id': parent_id,
                'persona_name': persona_data['name'],
                'persona_data': persona_data,
                'created_at': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.supabase.table('child_personas').insert(data).execute()
            return (True, "Persona saved successfully") if result.data else (False, "Failed to save persona")
            
        except Exception as e:
            print(f"Error saving persona: {e}")
            traceback.print_exc()
            return False, str(e)
    
    def load_persona_selector():
        """Display enhanced persona selector with quick-load functionality"""
        st.markdown("<h3>Saved Personas</h3>", unsafe_allow_html=True)
        
        if not st.session_state.get('parent_name'):
            st.warning("Please log in to access saved personas")
            return None
            
        success, personas = persona_manager.load_personas(st.session_state['parent_name'])
        
        if not success or not personas:
            st.info("No saved personas found. Create one by pressing the button below!")
            return None

    
    def update_persona(self, persona_id: str, updated_data: dict) -> Tuple[bool, str]:
        """Update an existing persona"""
        try:
            success, msg = self.validate_persona_data(updated_data)
            if not success:
                return False, msg
                
            result = self.supabase.supabase.table('child_personas')\
                .update({
                    "persona_data": updated_data,
                    "updated_at": datetime.utcnow().isoformat()
                })\
                .eq('id', persona_id)\
                .execute()
                
            return (True, "Persona updated successfully") if result.data else (False, "Failed to update persona")
            
        except Exception as e:
            print(f"Error updating persona: {e}")
            traceback.print_exc()
            return False, str(e)
    
    def delete_persona(self, persona_id: str) -> Tuple[bool, str]:
        """Delete a persona"""
        try:
            result = self.supabase.supabase.table('child_personas')\
                .delete()\
                .eq('id', persona_id)\
                .execute()
                
            return (True, "Persona deleted successfully") if result.data else (False, "Failed to delete persona")
            
        except Exception as e:
            print(f"Error deleting persona: {e}")
            traceback.print_exc()
            return False, str(e)

    def get_persona_by_id(self, persona_id: str) -> Tuple[bool, Optional[dict]]:
        """Retrieve a specific persona by ID"""
        try:
            result = self.supabase.supabase.table('child_personas')\
                .select("*")\
                .eq('id', persona_id)\
                .limit(1)\
                .execute()
                
            if result.data:
                return True, result.data[0]
            return True, None
            
        except Exception as e:
            print(f"Error retrieving persona: {e}")
            traceback.print_exc()
            return False, None

    def duplicate_persona(self, persona_id: str, new_name: str) -> Tuple[bool, str]:
        """Create a copy of an existing persona with a new name"""
        try:
            success, original = self.get_persona_by_id(persona_id)
            if not success or not original:
                return False, "Original persona not found"
            
            new_data = original['persona_data'].copy()
            new_data['name'] = new_name
            
            return self.save_persona(
                parent_id=original['parent_id'],
                persona_data=new_data
            )
            
        except Exception as e:
            print(f"Error duplicating persona: {e}")
            traceback.print_exc()
            return False, str(e)

    def validate_persona_data(self, persona_data: dict) -> Tuple[bool, str]:
            """Validate persona data structure and required fields (response_length removed)"""
            required_fields = ['name', 'communication_style', 'emotion_style']  # Removed 'response_length'
            try:
                missing_fields = [f for f in required_fields if not persona_data.get(f)]
                if missing_fields:
                    return False, f"Missing required fields: {', '.join(missing_fields)}"
                if not isinstance(persona_data.get('behaviors', []), list):
                    return False, "Behaviors must be a list"
                return True, "Persona data valid"
            except Exception as e:
                return False, f"Validation error: {str(e)}"
            
class SimulationAnalytics:
    def __init__(self, supabase_manager):
        self.supabase = supabase_manager
        
    def start_simulation(self, user_id: str) -> Tuple[bool, Optional[str]]:
        """Initialize a new simulation session"""
        try:
            simulation_data = {
                "user_id": user_id,
                "simulation_data": {
                    "start_time": datetime.utcnow().isoformat(),
                    "strategy_usage": {},
                    "response_times": [],
                    "emotional_trajectory": [],
                    "interaction_patterns": []
                },
                "created_at": datetime.utcnow().isoformat()
            }
            
            success, simulation_id = self.supabase.save_simulation_data(simulation_data)
            if success:
                return True, simulation_id
            return False, None
            
        except Exception as e:
            print(f"Error starting simulation: {e}")
            traceback.print_exc()
            return False, None

    def update_simulation_metrics(self, simulation_id: str, metrics: dict) -> bool:
        """Update metrics for an ongoing simulation"""
        try:
            if not simulation_id:
                return False
                
            data = {
                "simulation_id": simulation_id,
                "metrics": metrics,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.supabase.table('simulation_analytics')\
                .insert(data)\
                .execute()
                
            return bool(result.data)
            
        except Exception as e:
            print(f"Error updating simulation metrics: {e}")
            traceback.print_exc()
            return False

    def complete_simulation(self, simulation_id: str, final_metrics: dict) -> bool:
        """Mark simulation as completed and save final metrics"""
        try:
            # Update simulation completion status
            success, _ = self.supabase.complete_simulation(simulation_id)
            if not success:
                return False
            
            # Save final analytics
            data = {
                "simulation_id": simulation_id,
                "metrics": final_metrics,
                "is_final": True,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.supabase.table('simulation_analytics')\
                .insert(data)\
                .execute()
                
            return bool(result.data)
            
        except Exception as e:
            print(f"Error completing simulation: {e}")
            traceback.print_exc()
            return False

    def calculate_simulation_metrics(messages: list) -> dict:
        """Calculate detailed metrics from simulation messages."""
        try:
            total_exchanges = 0
            strategy_usage = {}
            response_times = []
            effective_approaches = 0
            growth_areas = 0
            conversation_timeline = []

            prev_time = None
            for msg in messages:
                # Record timeline regardless of role.
                conversation_timeline.append({
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "timestamp": msg.get("timestamp")
                })
                if msg['role'] == 'parent':
                    total_exchanges += 1
                    if 'strategy_used' in msg:
                        strategy = msg['strategy_used']
                        strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
                    if 'timestamp' in msg:
                        current_time = datetime.fromisoformat(msg['timestamp'])
                        if prev_time:
                            response_times.append((current_time - prev_time).total_seconds())
                        prev_time = current_time
                    if 'feedback' in msg:
                        if msg['feedback'].get('hint'):
                            effective_approaches += 1
                        if msg['feedback'].get('detailed'):
                            growth_areas += 1

            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            most_used_strategy = "None"
            if strategy_usage:
                most_used_strategy = max(strategy_usage.items(), key=lambda x: x[1])[0]
            
            detailed_metrics = {
                "total_exchanges": total_exchanges,
                "average_response_time": avg_response_time,
                "most_used_strategy": most_used_strategy,
                "strategy_usage": strategy_usage if strategy_usage else "No strategy usage data available",
                "effective_approaches": effective_approaches,
                "growth_areas": growth_areas,
                "conversation_timeline": conversation_timeline
            }
            
            return detailed_metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            traceback.print_exc()
            return {}

def track_simulation_progress(analytics: SimulationAnalytics, messages: list) -> None:
    """
    Track simulation progress by updating the simulation record with the current
    conversation chat (from messages) and then saving current metrics to simulation_analytics.
    """
    try:
        # If no simulation exists, start a new simulation.
        if not st.session_state.get('current_simulation_id'):
            simulation_data = {
                "start_time": datetime.utcnow().isoformat(),
                "strategy_usage": {},
                "response_times": [],
                "emotional_trajectory": [],
                "interaction_patterns": [],
                "conversation_history": []  # New key for the chat
            }
            
            result = supabase_manager.supabase.table('simulations').insert({
                "user_id": st.session_state.get('parent_name'),
                "simulation_data": simulation_data,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            
            if result.data:
                simulation_id = result.data[0].get('id')
                st.session_state['current_simulation_id'] = simulation_id
                # Save simulation_data in session for later updates.
                st.session_state['simulation_data'] = simulation_data
            else:
                print("Failed to start simulation tracking")
                return

        # Update conversation_history in simulation_data.
        current_conversation = st.session_state.conversation_branches[st.session_state.current_branch]
        simulation_data = st.session_state.get('simulation_data', {})
        simulation_data["conversation_history"] = current_conversation
        
        update_result = supabase_manager.supabase.table('simulations').update({
            "simulation_data": simulation_data
        }).eq("id", st.session_state['current_simulation_id']).execute()
        if not update_result.data:
            print("Failed to update conversation history in simulation record")
        
        # Now, calculate current metrics.
        current_metrics = {
            'total_exchanges': 0,
            'strategy_usage': {},
            'response_times': [],
            'feedback_stats': {
                'positive': 0,
                'constructive': 0
            }
        }
        
        prev_time = None
        for msg in messages:
            if msg.get('role') == 'parent':
                current_metrics['total_exchanges'] += 1
                
                if 'strategy_used' in msg:
                    strat = msg['strategy_used']
                    current_metrics['strategy_usage'][strat] = current_metrics['strategy_usage'].get(strat, 0) + 1
                
                if 'timestamp' in msg:
                    current_time = datetime.fromisoformat(msg['timestamp'])
                    if prev_time:
                        response_time = (current_time - prev_time).total_seconds()
                        current_metrics['response_times'].append(response_time)
                    prev_time = current_time
                
                if 'feedback' in msg:
                    if msg['feedback'].get('hint'):
                        current_metrics['feedback_stats']['positive'] += 1
                    if msg['feedback'].get('detailed'):
                        current_metrics['feedback_stats']['constructive'] += 1

        # Save analytics to Supabase.
        analytics_data = {
            'simulation_id': st.session_state['current_simulation_id'],
            'metrics': current_metrics,
            'created_at': datetime.utcnow().isoformat(),
            'is_final': False
        }
        
        result = supabase_manager.supabase.table('simulation_analytics').insert(analytics_data).execute()
        if not result.data:
            print("Failed to save analytics")
            
    except Exception as e:
        print(f"Error tracking simulation progress: {e}")
        traceback.print_exc()



def handle_simulation_completion(analytics: SimulationAnalytics) -> None:
    """Handle simulation completion and save final analytics"""
    try:
        simulation_id = st.session_state.get('current_simulation_id')
        if not simulation_id:
            return
            
        # Calculate final metrics
        messages = st.session_state.conversation_branches[st.session_state.current_branch]
        final_metrics = {
            'total_exchanges': len([m for m in messages if m['role'] == 'parent']),
            'strategy_usage': {},
            'response_times': [],
            'feedback_stats': {'positive': 0, 'constructive': 0},
            'completion_time': datetime.utcnow().isoformat(),
            'total_duration': (
                datetime.utcnow() - 
                datetime.fromisoformat(st.session_state.simulation_start_time)
            ).total_seconds()
        }
        
        # Calculate detailed metrics
        prev_time = None
        for msg in messages:
            if msg['role'] == 'parent':
                if 'strategy_used' in msg:
                    strategy = msg['strategy_used']
                    final_metrics['strategy_usage'][strategy] = \
                        final_metrics['strategy_usage'].get(strategy, 0) + 1
                        
                if 'timestamp' in msg:
                    current_time = datetime.fromisoformat(msg['timestamp'])
                    if prev_time:
                        response_time = (current_time - prev_time).total_seconds()
                        final_metrics['response_times'].append(response_time)
                    prev_time = current_time
                    
                if 'feedback' in msg:
                    if msg['feedback'].get('hint'):
                        final_metrics['feedback_stats']['positive'] += 1
                    if msg['feedback'].get('detailed'):
                        final_metrics['feedback_stats']['constructive'] += 1
        
        # Save final analytics
        analytics_data = {
            'simulation_id': simulation_id,
            'metrics': final_metrics,
            'created_at': datetime.utcnow().isoformat(),
            'is_final': True
        }
        
        analytics_result = supabase_manager.supabase.table('simulation_analytics')\
            .insert(analytics_data)\
            .execute()
            
        # Update simulation as completed
        simulation_result = supabase_manager.supabase.table('simulations')\
            .update({"completed_at": datetime.utcnow().isoformat()})\
            .eq('id', simulation_id)\
            .execute()
            
        if analytics_result.data and simulation_result.data:
            print(f"Successfully completed simulation {simulation_id}")
            st.session_state['current_simulation_id'] = None
        else:
            print("Failed to save completion analytics")
            
    except Exception as e:
        print(f"Error handling simulation completion: {e}")
        traceback.print_exc()

# Initialize managers
supabase_manager = SupabaseManager()
persona_manager = PersonaManager(supabase_manager)
analytics_manager = SimulationAnalytics(supabase_manager)


def display_persona_wizard():
    st.markdown("<h2 class='section-header'>Create Child Persona</h2>", unsafe_allow_html=True)
    
    # Initialize wizard state if not exists
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    if 'wizard_data' not in st.session_state:
        st.session_state.wizard_data = {
            'name': '',
            'communication_style': '',
            'emotion_style': 'Balanced',
            'response_length': 'Medium',
            'behaviors': [],
            'created_at': datetime.utcnow().isoformat()
        }

    # Progress indicator
    total_steps = 4
    progress = (st.session_state.wizard_step / total_steps) * 100
    
    st.markdown(f"""
        <div class="step-indicator">
            <div class="progress-bar-container">
                <div class="progress-bar" style="width: {progress}%"></div>
            </div>
            <div class="step-labels">
                <span>Step {st.session_state.wizard_step} of {total_steps}</span>
                <span>{int(progress)}% Complete</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        # Step 1: Basic Information
        if st.session_state.wizard_step == 1:
            st.markdown("### Name Your Persona Profile")
            st.markdown("""
                <div class="helper-text">
                    Create a name that helps you identify different behavior patterns or moods
                </div>
            """, unsafe_allow_html=True)
            
            profile_name = st.text_input(
                "Profile Name",
                value=st.session_state.wizard_data.get('name', ''),
                placeholder="e.g., After School Mood",
                help="Give this persona a distinctive name"
            )
            
            st.session_state.wizard_data['name'] = profile_name

        # Step 2: Communication Style
        elif st.session_state.wizard_step == 2:
            st.markdown("### Communication Style")
            st.markdown("""
                <div class="helper-text">
                    Describe how your child typically communicates in different situations
                </div>
            """, unsafe_allow_html=True)
            
            communication_style = st.text_area(
                "Typical Communication",
                value=st.session_state.wizard_data.get('communication_style', ''),
                placeholder="How do they express themselves? What phrases do they use often?",
                height=150
            )
            
            st.session_state.wizard_data['communication_style'] = communication_style

            # Preview card
            if communication_style:
                st.markdown("""
                    <div class="preview-card">
                        <div class="preview-label">Communication Style Preview</div>
                        <div class="preview-content">
                            {communication_style}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Step 3: Emotional Expression
        elif st.session_state.wizard_step == 3:
            st.markdown("### Emotional Expression")
            col1, col2 = st.columns([3, 2])
            
            with col1:
                emotion_options = ["Very Reserved", "Somewhat Reserved", "Balanced", 
                                 "Somewhat Expressive", "Very Expressive"]
                emotion_style = st.radio(
                    "How does your child typically express emotions?",
                    emotion_options,
                    index=emotion_options.index(
                        st.session_state.wizard_data.get('emotion_style', 'Balanced')
                    ),
                    horizontal=True
                )
                st.session_state.wizard_data['emotion_style'] = emotion_style
            
            with col2:
                st.markdown("""
                    <div class="preview-card">
                        <div class="preview-label">Selected Style</div>
                        <div class="preview-value">{}</div>
                    </div>
                """.format(emotion_style), unsafe_allow_html=True)

        # Step 4: Common Behaviors
        elif st.session_state.wizard_step == 4:
            st.markdown("### Common Behaviors")
            st.markdown("""
                <div class="helper-text">
                    Select behaviors that match your child's typical communication patterns
                </div>
            """, unsafe_allow_html=True)
            
            behavior_options = {
                "Communication": [
                    ("argues_frequently", "Argues and debates frequently", "💭"),
                    ("asks_questions", "Asks many questions", "❓"),
                    ("minimal_responses", "Gives minimal responses", "🤐")
                ],
                "Emotional": [
                    ("quiet_when_upset", "Becomes quiet when upset", "😶"),
                    ("easily_frustrated", "Shows frustration quickly", "😤"),
                    ("seeks_comfort", "Seeks reassurance", "🤗")
                ],
                "Interaction": [
                    ("physically_expressive", "Physically expressive", "👐"),
                    ("changes_subject", "Changes subject often", "↪️"),
                    ("takes_time", "Takes time to process", "⏳")
                ]
            }

            selected_behaviors = st.session_state.wizard_data.get('behaviors', [])
            
            for category, behaviors in behavior_options.items():
                st.subheader(category)
                cols = st.columns(len(behaviors))
                for i, (behavior_id, label, icon) in enumerate(behaviors):
                    with cols[i]:
                        if st.button(
                            f"{icon} {label}",
                            key=f"behavior_{behavior_id}",
                            type="primary" if behavior_id in selected_behaviors else "secondary",
                            help=f"Click to {'remove' if behavior_id in selected_behaviors else 'add'} this behavior",
                            use_container_width=True
                        ):
                            if behavior_id in selected_behaviors:
                                selected_behaviors.remove(behavior_id)
                            else:
                                selected_behaviors.append(behavior_id)
                            st.session_state.wizard_data['behaviors'] = selected_behaviors
                            st.rerun()

    # Navigation buttons:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.wizard_step > 1:
            if st.button("← Back", key="back_button", type="secondary", use_container_width=True):
                st.session_state.wizard_step -= 1
                st.rerun()
    with col3:
        if st.session_state.wizard_step < 4:
            if st.button("Next →", key="next_button", type="primary", use_container_width=True):
                # Validate current step
                if st.session_state.wizard_step == 1 and not st.session_state.wizard_data.get('name'):
                    st.error("Please provide a name for this persona")
                    return
                elif st.session_state.wizard_step == 2 and not st.session_state.wizard_data.get('communication_style'):
                    st.error("Please describe the communication style")
                    return
                st.session_state.wizard_step += 1
                st.rerun()
        else:
            if st.button("Save Persona", key="save_persona", type="primary", use_container_width=True):
                success = save_persona()  # Your existing save_persona() function call
                if success:
                    name = st.session_state.wizard_data.get('name', 'New Persona')
                    st.success(f"Persona '{name}' saved successfully!")
                    
                    # Immediately copy the wizard data (including behavior tags) into simulation state:
                    st.session_state.child_persona = st.session_state.wizard_data.copy()
                    st.session_state.selected_behavior_tags = st.session_state.wizard_data.get('behaviors', [])
                    
                    # If behavior tags were already selected, jump directly to conversation
                    if st.session_state.selected_behavior_tags:
                        st.session_state.simulation_stage = 'conversation'
                    else:
                        st.session_state.simulation_stage = 'setup'
                    
                    # Clear the wizard state and close the wizard
                    st.session_state.wizard_step = 1
                    st.session_state.wizard_data = {}
                    st.session_state['show_persona_wizard'] = False
                    
                    # Force role-play mode so the main UI defaults to Role-Play Simulation
                    st.session_state['role_play_active'] = True
                    time.sleep(1)
                    st.rerun()


def save_persona() -> bool:
    """Save the current persona to the database and update session state so that the simulation loads immediately."""
    if not st.session_state.get('parent_name'):
        st.error("Please log in to save personas")
        return False

    try:
        wizard_data = st.session_state.get('wizard_data', {})
        # Validate required fields
        if not wizard_data.get('name'):
            st.error("Please provide a name for this persona")
            return False
        if not wizard_data.get('communication_style'):
            st.error("Please describe the communication style")
            return False

        success, result = persona_manager.save_persona(
            parent_id=st.session_state['parent_name'],
            persona_data=wizard_data
        )

        if success:
            # Copy the wizard data into session state and preserve behavior tags (if any)
            st.session_state.child_persona = wizard_data.copy()
            if wizard_data.get('behaviors'):
                st.session_state.selected_behavior_tags = wizard_data['behaviors']
            # Force simulation stage to 'conversation' so that the behavior tag step is skipped
            st.session_state.simulation_stage = 'conversation'
            st.session_state.wizard_step = 1
            st.session_state.wizard_data = {}
            st.session_state['show_persona_wizard'] = False
            st.session_state['role_play_active'] = True  # Flag that we came from role-play
            time.sleep(1)
            st.rerun()
            return True

        st.error(f"Failed to save persona: {result}")
        return False

    except Exception as e:
        st.error(f"Error saving persona: {str(e)}")
        print(f"Detailed error: {str(e)}")
        traceback.print_exc()
        return False

def load_persona_selector():
    """Display enhanced persona selector with quick-load functionality"""
    st.markdown("<h3>Saved Personas</h3>", unsafe_allow_html=True)
    
    if not st.session_state.get('parent_name'):
        st.warning("Please log in to access saved personas")
        return None
            
    success, personas = persona_manager.load_personas(st.session_state['parent_name'])
    
    if not success or not personas:
        st.info("No saved personas found. Create one by clicking the button below!")
        return None

    # Display personas in a grid
    for persona in personas:
        with st.container():
            st.markdown(f"""
                <div class="persona-card">
                    <div class="persona-header">
                        <span class="persona-icon">👤</span>
                        <span class="persona-name">{persona['persona_name']}</span>
                    </div>
                    <div class="persona-details">
                        <p>Style: {persona['persona_data'].get('emotion_style', 'Not specified')}</p>
                        <p>Created: {datetime.fromisoformat(persona['created_at']).strftime('%B %d, %Y')}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(
                    "Load Persona",
                    key=f"load_persona_{persona['id']}",
                    help=f"Load {persona['persona_name']} persona",
                    use_container_width=True
                ):
                    st.session_state.child_persona = persona['persona_data']
                    st.success(f"Loaded persona: {persona['persona_name']}")
                    update_child_mood(persona['persona_data'])
                    time.sleep(0.5)
                    st.rerun()
            
            with col2:
                if st.button(
                    "🗑️",
                    key=f"delete_persona_{persona['id']}",
                    help=f"Delete {persona['persona_name']} persona",
                    use_container_width=True
                ):
                    success, msg = persona_manager.delete_persona(persona['id'])
                    if success:
                        st.success(f"Deleted persona: {persona['persona_name']}")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"Failed to delete persona: {msg}")

def try_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            params = st.query_params()
            params["__rerun__"] = str(random.random())
            st.query_params(**params)
        except Exception as e:
            st.error("Unable to rerun the app; please update your Streamlit version.")
            print("Rerun error:", e)


def simulate_conversation_streamlit(name: str, child_age: str, situation: str):
    """
    Complete simulation function with integrated analytics tracking and user interaction.
    """
    # Initialize analytics manager if not exists
    if 'analytics_manager' not in st.session_state:
        st.session_state.analytics_manager = SimulationAnalytics(supabase_manager)

    # Initialize simulation metrics if not exists
    if 'simulation_metrics' not in st.session_state:
        st.session_state.simulation_metrics = {
            'total_exchanges': 0,
            'strategy_usage': {},
            'response_times': [],
            'feedback_stats': {'positive': 0, 'constructive': 0},
            'emotional_trajectory': []
        }

    # If no persona exists, prompt the user to create one
    if not st.session_state.get('child_persona'):
        st.info("To begin, please create a persona for your child")
        if st.button("➕ Create New Persona", key="create_new_persona", use_container_width=True):
            st.session_state['show_persona_wizard'] = True
            st.session_state['role_play_active'] = True
            time.sleep(0.5)
            try_rerun()
        return

    # Initialize simulation if needed
    if st.session_state.get('simulation_stage') not in ['conversation', 'review']:
        # Set behavior tags from persona if available
        if st.session_state.child_persona.get('behaviors'):
            st.session_state.selected_behavior_tags = st.session_state.child_persona['behaviors']
        
        st.session_state.simulation_stage = 'conversation'
        
        # Start new simulation session
        success, simulation_id = st.session_state.analytics_manager.start_simulation(
            st.session_state.get('parent_name')
        )
        if success:
            st.session_state['current_simulation_id'] = simulation_id
            st.session_state['simulation_start_time'] = datetime.utcnow().isoformat()
        try_rerun()

    # --- Conversation Stage ---
    if st.session_state.simulation_stage == 'conversation':
        # Display simulation explanation
        st.markdown("""
        ### How the Role-Play Simulation Works
        1. **Create Your Child's Persona:** Create a persona that matches your child's communication style.
        2. **Choose a Strategy:** Select the communication strategy you want to practice.
        3. **Practice in Chat:** Engage in a simulated conversation.
        4. **Get Feedback:** Receive immediate feedback on your responses.
        """, unsafe_allow_html=True)

        # Strategy selection
        st.markdown("### Choose Communication Strategy")
        strategy = st.selectbox(
            "Select a strategy to practice:",
            options=["Active Listening", "Positive Reinforcement", "Reflective Questioning"],
            key="current_strategy",
            help="Select a communication strategy"
        )
        
        if strategy in STRATEGY_EXPLANATIONS:
            st.markdown(STRATEGY_EXPLANATIONS[strategy], unsafe_allow_html=True)

        # Active persona banner
        if st.session_state.get('child_persona'):
            st.markdown(f"""
                <div class="active-persona-banner">
                    <div class="persona-info">
                        <span class="persona-label">Active Persona:</span>
                        <span class="persona-name">{st.session_state.child_persona.get('name', 'Unnamed')}</span>
                        <span class="persona-style">({st.session_state.child_persona.get('emotion_style', 'Balanced')})</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Display chat interface
        st.markdown("### Practice Conversation")
        display_chat_ui(
            st.session_state.get('child_name', 'Child'),
            st.session_state.conversation_branches[st.session_state.current_branch],
            strategy
        )

        # Track current progress
        track_simulation_progress(
            st.session_state.analytics_manager,
            st.session_state.conversation_branches[st.session_state.current_branch]
        )

        # User input area
        user_input = st.text_area(
            "Your response:",
            key="chat_input",
            height=100,
            placeholder="Type your message here...",
            value=st.session_state.get("chat_input", "")
        )

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            send_button = st.button("💬 Send Response", key="send_response", type="primary", use_container_width=True)
        with col2:
            end_button = st.button("🎯 End Practice", key="end_practice", type="secondary", use_container_width=True)

        # Handle send response
        if send_button and user_input.strip():
            try:
                # Record start time for response timing
                response_start_time = datetime.utcnow()
                
                # Generate feedback and response
                feedback = provide_realtime_feedback(
                    user_input, strategy, situation, child_age,
                    st.session_state.conversation_branches[st.session_state.current_branch]
                )
                
                child_response = generate_child_response(
                    st.session_state.conversation_branches[st.session_state.current_branch],
                    child_age, situation, strategy, user_input
                )
                
                # Calculate response time
                response_time = (datetime.utcnow() - response_start_time).total_seconds()

                # Update simulation metrics
                st.session_state.simulation_metrics['total_exchanges'] += 1
                st.session_state.simulation_metrics['response_times'].append(response_time)
                st.session_state.simulation_metrics['strategy_usage'][strategy] = \
                    st.session_state.simulation_metrics['strategy_usage'].get(strategy, 0) + 1
                
                if feedback.get('hint'):
                    st.session_state.simulation_metrics['feedback_stats']['positive'] += 1
                if feedback.get('detailed'):
                    st.session_state.simulation_metrics['feedback_stats']['constructive'] += 1
                
                # Add messages to conversation
                st.session_state.conversation_branches[st.session_state.current_branch].extend([
                    {
                        "role": "parent",
                        "content": user_input,
                        "feedback": feedback,
                        "strategy_used": strategy,
                        "timestamp": datetime.utcnow().isoformat(),
                        "response_time": response_time
                    },
                    {
                        "role": "child",
                        "content": child_response,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ])
                
                # Update analytics
                track_simulation_progress(
                    st.session_state.analytics_manager,
                    st.session_state.conversation_branches[st.session_state.current_branch]
                )
                
                # Clear input and refresh
                if "chat_input" in st.session_state:
                    del st.session_state["chat_input"]
                try_rerun()
                
            except Exception as e:
                st.error("Error processing your response. Please try again.")
                print(f"Conversation error: {str(e)}")
                traceback.print_exc()

        # Handle end practice
        if end_button:
            try:
                # Handle simulation completion and analytics
                handle_simulation_completion(st.session_state.analytics_manager)
                # Update session state
                st.session_state.simulation_stage = 'review'
                try_rerun()
                
            except Exception as e:
                st.error("Error ending practice session. Please try again.")
                print(f"End practice error: {str(e)}")
                traceback.print_exc()

    # --- Review Stage ---
    elif st.session_state.simulation_stage == 'review':
        st.success("Practice session completed! Let's review your conversation.")
        
        # Display conversation review with current messages
        display_conversation_review(
            st.session_state.conversation_branches[st.session_state.current_branch]
        )

def display_conversation_review(messages: list):
    """Display detailed feedback from the conversation practice."""
    st.markdown("<h3>Conversation Practice Analysis</h3>", unsafe_allow_html=True)

    total_exchanges = sum(1 for msg in messages if msg['role'] == 'parent')
    strategy_usage = {}
    all_hints = []
    all_suggestions = []

    for msg in messages:
        if msg['role'] == 'parent':
            strategy = msg.get('strategy_used')
            if strategy:
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            if msg.get('feedback'):
                if msg['feedback'].get('hint'):
                    all_hints.append({
                        'feedback': msg['feedback']['hint'],
                        'response': msg['content']
                    })
                if msg['feedback'].get('detailed'):
                    all_suggestions.append({
                        'feedback': msg['feedback']['detailed'],
                        'response': msg['content']
                    })

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Exchanges", total_exchanges)
    with col2:
        most_used = max(strategy_usage.items(), key=lambda x: x[1])[0] if strategy_usage else "None"
        st.metric("Most Used Strategy", most_used)

    st.markdown("### 💪 What Worked Well")
    for item in all_hints:
        st.markdown(f"""
            <div class="feedback-detail" style="border-left: 4px solid #059669">
                <strong style="color: #059669">✓ {item['feedback']}</strong>
                <div style="margin-top: 8px; color: #666">
                    Example response: "{item['response']}"
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎯 Areas for Improvement")
    for item in all_suggestions:
        st.markdown(f"""
            <div class="feedback-detail" style="border-left: 4px solid #D97706">
                <strong style="color: #D97706">💡 {item['feedback']}</strong>
                <div style="margin-top: 8px; color: #666">
                    Related to response: "{item['response']}"
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Callback logic for the Start New Practice button:
    if st.button("🔄 Start New Practice", key="start_new_practice", type="primary", use_container_width=True):
        # Save analytics to Supabase before resetting
        if st.session_state.get('current_simulation_id'):
            metrics = {
                'total_exchanges': total_exchanges,
                'strategy_usage': strategy_usage,
                'completion_time': datetime.utcnow().isoformat(),
                'completed': True
            }
            analytics_data = {
                'simulation_id': st.session_state['current_simulation_id'],
                'metrics': metrics,
                'created_at': datetime.utcnow().isoformat(),
                'is_final': True
            }
            
            # Save to Supabase
            try:
                supabase_manager.supabase.table('simulation_analytics')\
                    .insert(analytics_data)\
                    .execute()
                    
                supabase_manager.supabase.table('simulations')\
                    .update({"completed_at": datetime.utcnow().isoformat()})\
                    .eq('id', st.session_state['current_simulation_id'])\
                    .execute()
            except Exception as e:
                print(f"Error saving analytics: {e}")
                traceback.print_exc()

        # Reset state
        st.session_state.conversation_branches = {0: []}
        st.session_state.current_branch = 0
        st.session_state.simulation_stage = 'conversation'
        st.session_state.pop('chat_input', None)
        time.sleep(0.5)
        try_rerun()

def handle_chat_response(user_input: str, strategy: str, situation: str, child_age: str):
    """Handle chat response and feedback"""
    try:
        feedback = provide_realtime_feedback(
            user_input, strategy, situation, child_age,
            st.session_state.conversation_branches[st.session_state.current_branch]
        )
        
        child_response = generate_child_response(
            st.session_state.conversation_branches[st.session_state.current_branch],
            child_age, situation, strategy, user_input
        )
        
        # Add messages to conversation
        st.session_state.conversation_branches[st.session_state.current_branch].extend([
            {
                "role": "parent",
                "content": user_input,
                "feedback": feedback,
                "strategy_used": strategy,
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "role": "child",
                "content": child_response,
                "timestamp": datetime.utcnow().isoformat()
            }
        ])
    except Exception as e:
        st.error("An error occurred processing your response")
        print(f"Chat error: {str(e)}")
        traceback.print_exc()

def generate_child_response(conversation_history: list, child_age: str, 
                          situation: str, strategy: str, parent_input: str) -> str:
    """Generate a contextual response from the child"""
    try:
        messages = [
            {
                "role": "system",
                "content": f"""
                    You are simulating a child aged {child_age} in this situation: {situation}
                    Parent is using {strategy} as their communication strategy.
                    Respond as the child would, maintaining consistency with their persona.
                    
                    Keep responses:
                    - Age-appropriate
                    - Brief (1-2 sentences)
                    - Natural and conversational
                    - Emotionally authentic
                    - Consistent with the child's communication style
                """
            }
        ]
        
        # Add conversation history
        for msg in conversation_history[-2:]:  # Last 2 messages for context
            role = "assistant" if msg["role"] == "child" else "user"
            messages.append({"role": role, "content": msg["content"]})
            
        # Add current parent input
        messages.append({"role": "user", "content": parent_input})
        
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating child response: {e}")
        return "I don't know what to say..."

def end_practice_session():
    """Handle ending the practice session"""
    st.session_state.simulation_stage = 'review'
    if st.session_state.get('current_simulation_id'):
        save_simulation_analytics(
            supabase_manager,
            st.session_state.current_simulation_id,
            st.session_state.simulation_metrics
        )

def display_chat_ui(child_name: str, messages: list, strategy: str):
    """Display a chat interface with chat bubbles and immediate feedback."""
    st.markdown(f"<h4 style='margin-bottom:10px;'>Current Strategy: {strategy}</h4>", unsafe_allow_html=True)
    for msg in messages:
        if msg['role'] == 'parent':
            st.markdown(f"""
                <div class="message-wrapper message-right">
                    <div class="message parent-message">
                        <strong>You:</strong> {msg['content']}
                    </div>
                </div>
                {f'''
                <div class="feedback-bubble">
                    <div class="feedback-hint">{msg["feedback"]["hint"]}</div>
                </div>
                ''' if msg.get('feedback') else ''}
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="message-wrapper message-left">
                    <div class="message child-message">
                        <strong>{child_name}:</strong> {msg['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)



    

def provide_realtime_feedback(user_input: str, strategy: str, situation: str, child_age: str, conversation_history: list) -> dict:
    """Generate real-time feedback on parent responses"""
    try:
        messages = [
            {
                "role": "system",
                "content": f"""
                    Analyze parent response in a parenting conversation.
                    Context: Child age {child_age}, Situation: {situation}
                    Current strategy: {strategy}
                    
                    Provide feedback as JSON with:
                    - hint: Brief encouraging feedback highlighting what worked well (10-15 words)
                    - detailed: Growth-oriented suggestion for the review phase, focusing on opportunities to enhance the approach
                """
            },
            {
                "role": "user", 
                "content": f"Parent response: {user_input}"
            }
        ]
        
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        
        return json.loads(completion.choices[0].message.content)
        
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return {
            "hint": "Consider being more specific and empathetic in your response",
            "detailed": "Try incorporating active listening techniques and showing more understanding"
        }

def track_simulation_metrics(parent_message: str, child_response: str, strategy: str):
    """Save analytics to Supabase."""
    try:
        current_time = datetime.now()
        simulation_id = st.session_state.get('simulation_id')
        
        metrics = {
            'total_exchanges': len(st.session_state.conversation_branches[st.session_state.current_branch]),
            'strategy_usage': {strategy: 1},
            'response_time': (current_time - st.session_state.last_response_time).total_seconds() 
                             if hasattr(st.session_state, 'last_response_time') else 0
        }
        
        # Save metrics to Supabase
        data = {
            'simulation_id': simulation_id,
            'user_id': st.session_state.parent_name,
            'metrics': metrics,
            'created_at': current_time.isoformat()
        }
        
        supabase_manager.supabase.table('simulation_analytics').insert(data).execute()
        st.session_state.last_response_time = current_time
        
    except Exception as e:
        print(f"Error saving analytics: {str(e)}")
        traceback.print_exc()

def reset_simulation_state():
    """Reset simulation-specific session state variables"""
    simulation_vars = {
        'conversation_branches': {0: []},
        'current_branch': 0,
        'simulation_stage': 'setup',
        'selected_behavior_tags': [],
        'current_strategy': "Active Listening",
        'simulation_metrics': {
            'total_exchanges': 0,
            'response_times': [],
            'strategy_usage': {},
            'feedback_stats': {'positive': 0, 'constructive': 0},
            'emotional_trajectory': [],
            'interaction_patterns': [],
            'behavior_triggers': []
        },
        'simulation_start_time': datetime.utcnow().isoformat(),
        'last_interaction_time': datetime.utcnow(),
        'stored_responses': {},
        'final_metrics': None
    }
    
    for var, value in simulation_vars.items():
        st.session_state[var] = value
        
    # Update child mood if context is available
    if all(st.session_state.get(k) for k in ['child_persona', 'situation', 'child_age']):
        try:
            st.session_state['child_mood'] = determine_child_mood(
                st.session_state['child_persona'],
                st.session_state['situation'],
                st.session_state['child_age']
            )
        except Exception as e:
            print(f"Error resetting simulation mood: {e}")
            st.session_state['child_mood'] = 'neutral'

def display_advice(parent_name: str, child_age: str, situation: str):
    """Display enhanced parenting advice with state management and save functionality"""
    st.markdown("<h2 class='section-header'>Parenting Advice</h2>", unsafe_allow_html=True)
    
    if not situation:
        st.warning("Please describe the situation to get advice.")
        return

    try:
        # State management for advice
        if 'current_advice' not in st.session_state:
            with st.spinner('Generating advice...'):
                messages = [
                    {
                        "role": "system",
                        "content": f"""
                            You are a parenting expert. Generate 4 specific, practical pieces of advice for the given situation.
                            Each piece should be:
                            - Detailed enough to be actionable (25-30 words)
                            - Specific to the situation and child's age
                            - Based on evidence-backed parenting strategies
                            - Include both what to do and why it works

                            Format each piece as a JSON array of objects with:
                            - title: A short 2-3 word summary
                            - advice: The detailed advice
                            - icon: A relevant emoji
                            - color: Background color from these options:
                              ["#D9E9FF", "#C2FAD8", "#FFECC2", "#E8D9FF"]
                            - category: One of: ["Immediate Action", "Long-term Strategy", "Communication Tip", "Prevention"]
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Child age: {child_age}\nSituation: {situation}\nGenerate specific advice."
                    }
                ]

                completion = openai.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7
                )
                
                st.session_state.current_advice = json.loads(completion.choices[0].message.content)

        # Display advice using stored state
        advice_list = st.session_state.current_advice
        
        # Add refresh button
        col1, col2, col3 = st.columns([1, 4, 1])
        with col3:
            if st.button("🔄 Refresh", help="Generate new advice"):
                st.session_state.pop('current_advice', None)
                st.rerun()
                
        # Display advice cards in columns
        for i in range(0, len(advice_list), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(advice_list):
                    advice = advice_list[i + j]
                    
                    with cols[j]:
                        st.markdown(f"""
                            <div class="advice-card" 
                                 style="background-color: {advice['color']}">
                                <div class="advice-category-badge">
                                    {advice['category']}
                                </div>
                                <div class="advice-title">
                                    <span class="advice-icon">{advice['icon']}</span>
                                    {advice['title']}
                                </div>
                                <div class="advice-content">
                                    {advice['advice']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        save_key = f"save_advice_{i+j}"
                        if st.button("💾 Save Advice", key=save_key, type="secondary"):
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
                                        "color": advice['color'],
                                        "category": advice['category']
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

        # Add button to retry
        if st.button("🔄 Try Again"):
            st.session_state.pop('current_advice', None)
            st.rerun()

def display_communication_techniques(situation: str):
    """Display communication techniques with improved error handling and UI"""
    st.markdown("<h2 class='section-header'>Communication Techniques</h2>", unsafe_allow_html=True)
    
    if not situation:
        st.warning("Please describe the situation to get communication techniques.")
        return

    try:
        # State management for techniques
        if 'current_techniques' not in st.session_state:
            with st.spinner('Generating techniques...'):
                messages = [
                    {"role": "system", "content": """
                        Generate exactly 3 communication strategies as a JSON array.
                        Each strategy object must have:
                        {
                            "title": "Strategy name with emoji",
                            "purpose": "Single clear sentence",
                            "steps": ["step 1", "step 2", "step 3"],
                            "example": "Brief specific example",
                            "outcome": "Expected result",
                            "color": "One of: #7C3AED, #0D9488, #D97706"
                        }
                        Format as {"strategies": [strategy1, strategy2, strategy3]}
                    """},
                    {"role": "user", "content": f"Generate 3 strategies for this situation: {situation}"}
                ]
                
                try:
                    completion = openai.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    response_content = completion.choices[0].message.content
                    
                    # Parse and validate JSON response
                    try:
                        parsed = json.loads(response_content)
                        if 'strategies' not in parsed:
                            # If strategies key is missing, wrap the array
                            if isinstance(parsed, list):
                                parsed = {'strategies': parsed}
                            else:
                                raise ValueError("Invalid response structure")
                        
                        techniques = parsed['strategies']
                        if not isinstance(techniques, list) or len(techniques) == 0:
                            raise ValueError("No strategies found in response")
                            
                        # Validate each strategy
                        required_fields = ['title', 'purpose', 'steps', 'example', 'outcome', 'color']
                        for strategy in techniques:
                            missing = [f for f in required_fields if f not in strategy]
                            if missing:
                                raise ValueError(f"Strategy missing fields: {missing}")
                            if not isinstance(strategy['steps'], list):
                                raise ValueError("Steps must be an array")
                                
                        st.session_state.current_techniques = techniques
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error: {str(e)}\nResponse: {response_content}")
                        raise Exception("Invalid JSON format in response")
                    except ValueError as e:
                        print(f"Validation error: {str(e)}\nResponse: {response_content}")
                        raise Exception("Invalid response structure")
                        
                except Exception as e:
                    print(f"API or validation error: {str(e)}")
                    raise
        
        # Add refresh button with unique key
        col1, col2, col3 = st.columns([1, 4, 1])
        with col3:
            if st.button("🔄 Refresh", key="refresh_techniques", help="Generate new communication techniques"):
                st.session_state.pop('current_techniques', None)
                st.rerun()

        # Display techniques from state
        strategies = st.session_state.current_techniques
        
        cols = st.columns(3)
        for idx, (strategy, col) in enumerate(zip(strategies, cols)):
            with col:
                st.markdown(f"""
                    <div class="strategy-card" style="background: white;">
                        <div class="strategy-header" style="background-color: {strategy['color']};">
                            <h3>{strategy['title']}</h3>
                        </div>
                        <div class="strategy-content">
                            <div class="purpose-section">
                                {strategy['purpose']}
                            </div>
                            <div class="steps-section">
                                {"".join([f'<div class="step-item">• {step}</div>' for step in strategy['steps']])}
                            </div>
                            <div class="example-section">
                                <strong>Example:</strong><br>
                                {strategy['example']}
                            </div>
                            <div class="outcome-section">
                                <strong>Expected Outcome:</strong><br>
                                {strategy['outcome']}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Save button with unique key
                save_key = f"save_technique_{idx}"
                if st.button("💾 Save Technique", key=save_key, type="secondary"):
                    if not st.session_state.get('parent_name'):
                        st.warning("Please log in to save techniques")
                    else:
                        try:
                            success, error = supabase_manager.save_item_to_supabase(
                                parent_id=st.session_state['parent_name'],
                                item_type="technique",
                                title=strategy['title'],
                                content=f"{strategy['purpose']}\n\nSteps:\n" + 
                                       "\n".join([f"• {step}" for step in strategy['steps']]) +
                                       f"\n\nExample: {strategy['example']}\n\nOutcome: {strategy['outcome']}",
                                metadata={
                                    "color": strategy['color'],
                                    "steps": strategy['steps'],
                                    "outcome": strategy['outcome']
                                }
                            )
                            if success:
                                st.success("Technique saved successfully!")
                            else:
                                st.error(f"Failed to save: {error}")
                        except Exception as e:
                            st.error(f"Error saving technique: {str(e)}")
                            print(f"Save error details: {str(e)}")
                            traceback.print_exc()
    
    except Exception as e:
        print(f"Error displaying techniques: {str(e)}")
        traceback.print_exc()
        st.error("Unable to generate communication techniques. Please try again.")
        
        # Retry button with unique key
        if st.button("🔄 Try Again", key="retry_techniques"):
            st.session_state.pop('current_techniques', None)
            st.rerun()


def display_conversation_starters(situation: str):
    """Display conversation starters with state management"""
    st.markdown("<h2 class='section-header'>Conversation Starters</h2>", unsafe_allow_html=True)
    
    if not situation:
        st.warning("Please describe the situation to get conversation starters.")
        return

    try:
        # State management for conversation starters
        if 'current_starters' not in st.session_state:
            with st.spinner("Generating conversation starters..."):
                messages = [
                    {
                        "role": "system",
                        "content": f"""
                            Generate 5 conversation starters for talking with a child about this situation.
                            Each starter should:
                            - Be a complete question or statement
                            - Use age-appropriate language
                            - Encourage open dialogue
                            - Be specific to the situation

                            Format as JSON array with:
                            - text: The complete conversation starter
                            - category: From ["Feelings", "Understanding", "Activities", "Solutions", "Ideas"]
                            - approach: Brief description of why this approach works
                            - icon: Matching emoji
                            - color: From ["#EBF4FF", "#FEF3F2", "#F0FDF4", "#FDF2F8", "#FDF6B2"]
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Generate conversation starters for: {situation}"
                    }
                ]
                
                completion = openai.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7
                )
                
                st.session_state.current_starters = json.loads(completion.choices[0].message.content)

        # Add refresh button
        col1, col2, col3 = st.columns([1, 4, 1])
        with col3:
            if st.button("🔄 Refresh Starters", help="Generate new conversation starters"):
                st.session_state.pop('current_starters', None)
                st.rerun()

        # Display starters from state
        starters = st.session_state.current_starters
        
        for starter in starters:
            st.markdown(f"""
                <div class="starter-card" style="background-color: {starter['color']};">
                    <div class="starter-category">
                        <span class="category-icon">{starter['icon']}</span>
                        <span class="category-name">{starter['category']}</span>
                    </div>
                    <div class="starter-text">
                        "{starter['text']}"
                    </div>
                    <div class="starter-approach">
                        <em>Why this works:</em> {starter['approach']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            save_key = f"save_starter_{starters.index(starter)}"
            if st.button("💾 Save Starter", key=save_key, type="secondary"):
                if not st.session_state.get('parent_name'):
                    st.warning("Please log in to save conversation starters")
                else:
                    success, error = supabase_manager.save_item_to_supabase(
                        parent_id=st.session_state['parent_name'],
                        item_type="starter",
                        title=f"{starter['icon']} {starter['category']}",
                        content=f"{starter['text']}\n\nApproach: {starter['approach']}",
                        metadata={
                            "category": starter['category'],
                            "icon": starter['icon'],
                            "color": starter['color'],
                            "approach": starter['approach']
                        }
                    )
                    if success:
                        st.success("Conversation starter saved!")
                    else:
                        st.error(f"Failed to save: {error}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        st.error("Unable to generate conversation starters at this time.")
        
        if st.button("🔄 Try Again"):
            st.session_state.pop('current_starters', None)
            st.rerun()

def init_simulation_state():
    """Initialize simulation-specific variables"""
    if 'conversation_branches' not in st.session_state:
        st.session_state['conversation_branches'] = {0: []}
    if 'current_branch' not in st.session_state:
        st.session_state['current_branch'] = 0
    if 'simulation_stage' not in st.session_state:
        st.session_state['simulation_stage'] = 'setup'
    if 'selected_behavior_tags' not in st.session_state:
        st.session_state['selected_behavior_tags'] = []

def init_session_state():
    """Initialize all session state variables with proper defaults."""
    
    # Core session identifiers
    session_vars = {
        # User information
        'info_submitted': False,
        'parent_name': None,
        'child_name': None,
        'child_age': None,
        'situation': None,
        'prolific_id': None,
        'parent_info_id': None,
        
        # Simulation identifiers
        'run_id': str(uuid4()),
        'simulation_id': str(uuid4()),
        'current_simulation_id': None,
        'simulation_start_time': datetime.utcnow().isoformat(),
        
        # Enhanced persona management
        'show_persona_wizard': False,
        'wizard_step': 1,
        'wizard_data': {
            'name': '',
            'communication_style': '',
            'emotion_style': 'Balanced',
            'response_length': 'Medium',
            'behaviors': []
        },
        'child_persona': None,
        'saved_personas': {},
        
        # Simulation state
        'simulation_stage': 'setup',
        'selected_behavior_tags': [],
        'emotional_intensity': 50,
        'conversation_branches': {0: []},
        'current_branch': 0,
        'simulation_ended': False,
        'stored_responses': {},
        'chat_input': '',
        
        # Enhanced metrics tracking
        'simulation_metrics': {
            'total_exchanges': 0,
            'response_times': [],
            'strategy_usage': {},
            'feedback_stats': {
                'positive': 0,
                'constructive': 0
            },
            'emotional_trajectory': [],
            'interaction_patterns': [],
            'persona_influence': [],
            'behavior_triggers': []
        },
        
        # Final metrics for review
        'final_metrics': None,
        
        # UI state
        'show_tutorial': True,
        'show_saved_items': False,
        'current_strategy': "Active Listening",
        'show_feedback': True,
        'show_hints': True,
        'compact_view': False,
        'role_play_active': False,
        
        # Feature tracking
        'visited_features': set(),
        
        # Analytics
        'session_start_time': datetime.utcnow(),
        'last_interaction_time': datetime.utcnow(),
        'session_duration': 0,
        'interaction_count': 0,
        
        # Analytics manager
        'analytics_manager': None
    }
    
    # Initialize or update session state variables
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
            
    # Ensure conversation branches exist
    if not st.session_state.conversation_branches:
        st.session_state.conversation_branches = {0: []}
        
    # Initialize analytics manager if needed
    if (st.session_state.analytics_manager is None and 
        'supabase_manager' in globals()):
        st.session_state.analytics_manager = SimulationAnalytics(supabase_manager)
        
    # Update session duration
    if st.session_state.session_start_time:
        st.session_state.session_duration = (
            datetime.utcnow() - st.session_state.session_start_time
        ).total_seconds()


@st.cache_data(ttl=3600)
def cached_openai_call(messages, model="gpt-4", temperature=0.7, max_tokens=150):
    """Cache OpenAI API calls to improve performance"""
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

def generate_response_with_citations(prompt: str, citations: dict) -> str:
    """Generate responses using relevant citations"""
    try:
        messages = [
            {
                "role": "system",
                "content": "Use the provided academic citations and website resources to generate responses."
            },
            {
                "role": "user",
                "content": f"""
                Academic Citations:
                {citations.get('academic', '')}

                Website Resources:
                {citations.get('web', '')}

                Question: {prompt}
                """
            }
        ]
        
        return cached_openai_call(messages)
    except Exception as e:
        print(f"Error generating response with citations: {e}")
        return "Unable to generate response at this time."
    
def show_info_screen():
    """Display the initial information collection screen"""
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
            <div class='form-description'>
                Please enter your <b>Prolific ID</b> (24-character identifier from your Prolific account)
            </div>
        """, unsafe_allow_html=True)
        
        prolific_id = st.text_input(
            "Prolific ID", 
            value=parent_name,
            placeholder="Enter your 24-character Prolific ID...",
            help="This is the ID assigned to you by Prolific"
        )
        
        child_name = st.text_input(
            "Child's Name or Nickname", 
            value=child_name,
            placeholder="Enter your child's name or pseudonym..."
        )
        
        child_age = st.selectbox(
            "Child's Age Range",
            ["3-5 years", "6-9 years", "10-12 years"],
            index=["3-5 years", "6-9 years", "10-12 years"].index(child_age) if child_age else 0
        )
        
        situation = st.text_area(
            "Situation Description",
            value=situation,
            placeholder="Describe the parenting situation you'd like help with...",
            height=120
        )
        
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

def display_saved_items():
    """Display saved items with unique keys and improved error handling"""
    st.markdown("<h2>Your Saved Content</h2>", unsafe_allow_html=True)
    
    # Add back button at top
    if st.button("← Back to Main Menu", key="back_to_main", type="secondary"):
        st.session_state['show_saved_items'] = False
        st.rerun()
    
    if not st.session_state.get('parent_name'):
        st.warning("Please log in to view saved content")
        return
    
    # Add filter options
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        filter_type = st.multiselect(
            "Filter by type:",
            options=["advice", "technique", "starter"],
            default=["advice", "technique", "starter"]
        )
    
    try:
        success, items = supabase_manager.get_saved_items(st.session_state['parent_name'])
        
        if not success:
            st.error("Failed to retrieve saved items")
            return
            
        if not items:
            st.info("You haven't saved any content yet. Browse through the different sections and click 'Save' on items you want to keep!")
            return
        
        # Filter and display items
        filtered_items = [item for item in items if item['item_type'] in filter_type]
        
        for i, item in enumerate(filtered_items):
            with st.container():
                # Get styling information from metadata
                metadata = item.get('metadata', {})
                bg_color = metadata.get('color', '#4338CA')
                icon = metadata.get('icon', '📝')
                
                st.markdown(f"""
                    <div class="saved-item" style="background-color: {bg_color}">
                        <div class="item-header">
                            <span class="item-icon">{icon}</span>
                            <span class="item-title">{item['title']}</span>
                            <span class="item-type">{item['item_type'].title()}</span>
                        </div>
                        <div class="item-content">
                            {item['content']}
                        </div>
                        <div class="item-metadata">
                            Saved on: {datetime.fromisoformat(item['saved_at']).strftime('%B %d, %Y %I:%M %p')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Delete button with unique key
                delete_key = f"delete_item_{i}_{item['id']}"
                if st.button("🗑️ Delete", key=delete_key, type="secondary"):
                    try:
                        success, error = supabase_manager.delete_saved_item(item['id'])
                        if success:
                            st.success("Item deleted!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Failed to delete item: {error}")
                    except Exception as e:
                        st.error(f"Error deleting item: {str(e)}")
                        print(f"Delete error details: {str(e)}")
                        traceback.print_exc()
                
    except Exception as e:
        st.error("An error occurred while displaying saved items")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        
        # Add retry button
        if st.button("🔄 Retry", key="retry_load_items"):
            st.rerun()

def display_behavior_tags():
    """Display and handle behavior tag selection"""
    st.markdown("### Quick Behavior Tags")
    
    behavior_tags = {
        "Communication": [
            {"id": "argues_back", "label": "Argues Back", "icon": "💭", 
             "description": "Tends to debate or challenge rules"},
            {"id": "asks_questions", "label": "Asks Many Questions", "icon": "❓",
             "description": "Seeks clarification frequently"},
            {"id": "minimal_responses", "label": "Brief Responses", "icon": "🤐",
             "description": "Uses few words to answer"}
        ],
        "Emotional": [
            {"id": "shuts_down", "label": "Shuts Down", "icon": "🤫",
             "description": "Becomes quiet when upset"},
            {"id": "easily_frustrated", "label": "Easily Frustrated", "icon": "😤",
             "description": "Shows frustration quickly"},
            {"id": "seeks_comfort", "label": "Seeks Comfort", "icon": "🤗",
             "description": "Looks for reassurance"}
        ],
        "Attention": [
            {"id": "easily_distracted", "label": "Easily Distracted", "icon": "🦋",
             "description": "Shifts focus frequently"},
            {"id": "hyper_focused", "label": "Intensely Focused", "icon": "🎯",
             "description": "Gets absorbed in interests"},
            {"id": "needs_reminders", "label": "Needs Prompting", "icon": "⏰",
             "description": "Benefits from reminders"}
        ]
    }
    
    st.info("Select 2-3 tags that best describe your child's typical responses")
    
    for category, tags in behavior_tags.items():
        st.subheader(category)
        cols = st.columns(len(tags))
        for i, tag in enumerate(tags):
            with cols[i]:
                is_selected = tag['id'] in st.session_state.selected_behavior_tags
                button_type = "primary" if is_selected else "secondary"
                
                if st.button(
                    f"{tag['icon']} {tag['label']}\n{tag['description']}",
                    key=f"tag_{tag['id']}",
                    help=tag['description'],
                    type=button_type,
                    use_container_width=True
                ):
                    if is_selected:
                        st.session_state.selected_behavior_tags.remove(tag['id'])
                    elif len(st.session_state.selected_behavior_tags) < 3:
                        st.session_state.selected_behavior_tags.append(tag['id'])
                    st.rerun()

def determine_child_mood(persona: dict, situation: str, child_age: str) -> str:
    """Determines mood based on persona, situation and age"""
    try:
        # Get base weights from analysis
        persona_weights = analyze_child_persona(persona)
        situation_weights = analyze_situation_context(situation, child_age)
        
        # Combine weights with configurable emphasis
        final_weights = {
            mood: (p_weight * 0.6) + (s_weight * 0.4)
            for mood, (p_weight, s_weight) in 
            zip(persona_weights.keys(), zip(persona_weights.values(), situation_weights.values()))
        }
        
        # Normalize weights and select mood
        total = sum(final_weights.values())
        probabilities = {m: w/total for m, w in final_weights.items()}
        
        return random.choices(
            list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]
        
    except Exception as e:
        print(f"Error determining mood: {e}")
        return 'neutral'

def analyze_situation_context(situation: str, child_age: str) -> dict:
    """Analyze situation for mood indicators"""
    situation_lower = situation.lower()
    base_weights = {'cooperative': 1.0, 'defiant': 1.0, 'distracted': 1.0}
    
    # Analyze keywords in situation
    keywords = {
        'cooperative': ['play', 'help', 'share', 'together', 'calm', 'happy'],
        'defiant': ['no', 'won\'t', 'refuse', 'angry', 'upset', 'mad'],
        'distracted': ['busy', 'tired', 'many', 'lots', 'excited', 'overwhelmed']
    }
    
    for mood, words in keywords.items():
        matches = sum(1 for word in words if word in situation_lower)
        base_weights[mood] += matches * 0.2
    
    # Apply age-specific factors
    if child_age in MOOD_WEIGHTS['age_factors']:
        age_adjustments = MOOD_WEIGHTS['age_factors'][child_age]
        for mood, adjustment in age_adjustments.items():
            base_weights[mood] *= adjustment
    
    # Apply time-of-day factors if mentioned
    time_indicators = {
        'morning': ['morning', 'breakfast', 'wake'],
        'afternoon': ['afternoon', 'lunch', 'school'],
        'evening': ['evening', 'dinner', 'bedtime']
    }
    
    for time, indicators in time_indicators.items():
        if any(ind in situation_lower for ind in indicators):
            time_adjustments = MOOD_WEIGHTS['time_factors'][time]
            for mood, adjustment in time_adjustments.items():
                base_weights[mood] *= adjustment
    
    return base_weights

def analyze_child_persona(persona: dict) -> dict:
    """Analyze persona for baseline mood tendencies"""
    if not isinstance(persona, dict):
        return {'cooperative': 1.0, 'defiant': 1.0, 'distracted': 1.0}
    
    weights = {'cooperative': 1.0, 'defiant': 1.0, 'distracted': 1.0}
    
    # Analyze communication style
    style_text = ' '.join([
        str(persona.get('communication_style', '')),
        str(persona.get('typical_phrases', ''))
    ]).lower()
    
    # Communication patterns influence
    patterns = {
        'cooperative': ['listens', 'calm', 'helps', 'shares', 'patient'],
        'defiant': ['argues', 'refuses', 'angry', 'protests', 'no'],
        'distracted': ['changes subject', 'forgets', 'looks away', 'busy']
    }
    
    for mood, indicators in patterns.items():
        matches = sum(1 for pattern in indicators if pattern in style_text)
        weights[mood] += matches * 0.3
    
    # Emotional expression influence
    emotion_style = persona.get('emotion_style', 'Balanced')
    emotion_adjustments = {
        'Very Reserved': {'cooperative': 0.4, 'defiant': -0.2, 'distracted': 0.1},
        'Somewhat Reserved': {'cooperative': 0.2, 'defiant': -0.1},
        'Balanced': {},
        'Somewhat Expressive': {'defiant': 0.2, 'distracted': 0.2},
        'Very Expressive': {'defiant': 0.4, 'distracted': 0.3, 'cooperative': -0.2}
    }
    
    if emotion_style in emotion_adjustments:
        for mood, adjustment in emotion_adjustments[emotion_style].items():
            weights[mood] += adjustment
    
    # Behavior pattern influence
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
    
    behaviors = persona.get('behaviors', [])
    for mood, indicators in behavior_weights.items():
        matches = sum(1 for behavior in behaviors if behavior in indicators)
        weights[mood] += matches * 0.4
    
    return weights

def update_child_mood(persona: dict):
    """Updates child's mood based on persona and situation"""
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


def display_progress_sidebar(feature_order):
    """Display progress of features visited in sidebar"""
    if 'visited_features' in st.session_state:
        st.sidebar.markdown("### Your Progress")
        for feature in feature_order.keys():
            feature_key = feature.lower().replace(" ", "_")
            is_visited = (feature_key in st.session_state.visited_features or 
                        feature in st.session_state.visited_features)
            
            st.sidebar.markdown(f"""
                <div class="progress-item {'completed' if is_visited else ''}">
                    <span class="progress-icon">{'✅' if is_visited else '◽'}</span>
                    <span class="progress-text">{feature}</span>
                </div>
            """, unsafe_allow_html=True)

# Feature order and descriptions
feature_order = {
    "Advice": "Get expert guidance on handling specific parenting situations based on evidence-based strategies.",
    "Communication Techniques": "Discover helpful ways to communicate with your child and get tips on how to address your specific situation.",
    "Conversation Starters": "Receive help initiating difficult conversations with suggested opening phrases and questions.",
    "Role-Play Simulation": "Practice conversations in a safe environment to develop and refine your communication approach.",
}

# Mood Analysis Configuration
MOOD_WEIGHTS = {
    'age_factors': {
        "3-5 years": {
            'distracted': 1.3,
            'defiant': 1.2,
            'cooperative': 0.9
        },
        "6-9 years": {
            'cooperative': 1.2,
            'distracted': 1.1,
            'defiant': 1.0
        },
        "10-12 years": {
            'defiant': 1.2,
            'cooperative': 0.9,
            'distracted': 0.8
        }
    },
    'time_factors': {
        'morning': {'cooperative': 1.1, 'distracted': 0.9},
        'afternoon': {'distracted': 1.2, 'cooperative': 0.9},
        'evening': {'defiant': 1.2, 'cooperative': 0.8}
    }
}

def show_tutorial():
    """Display enhanced tutorial for first-time users"""
    st.markdown("# Welcome to the Parenting Support Bot! 🎉")
    
    st.markdown("""
        <div class="tutorial-container">
            <p class="tutorial-intro">
                This app is designed to help you develop effective parenting strategies through:
            </p>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">📚</div>
                    <div class="feature-title">Expert Advice</div>
                    <div class="feature-desc">Get evidence-based parenting advice</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🗣️</div>
                    <div class="feature-title">Communication Techniques</div>
                    <div class="feature-desc">Discover effective communication strategies</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">💭</div>
                    <div class="feature-title">Conversation Starters</div>
                    <div class="feature-desc">Learn how to begin difficult conversations</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🎮</div>
                    <div class="feature-title">Role-Play Simulation</div>
                    <div class="feature-desc">Practice conversations in a safe environment</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Got it, let's start!", use_container_width=True):
            st.session_state.show_tutorial = False
            st.rerun()

def update_session_analytics():
    if 'session_start_time' in st.session_state:
        current_time = datetime.now()
        st.session_state['session_duration'] = (current_time - st.session_state['session_start_time']).total_seconds()
        st.session_state['last_interaction_time'] = current_time
        
        # Check for session timeout (30 minutes)
        time_since_last_interaction = (current_time - st.session_state['last_interaction_time']).total_seconds()
        if time_since_last_interaction > 1800:
            st.warning("Your session has been inactive for 30 minutes. Please refresh the page to continue.")

def reset_session_info():
    """Reset user session information for re-entry"""
    keys_to_reset = [
        'info_submitted',
        'parent_name',
        'child_name',
        'child_age',
        'situation',
        'prolific_id',
        'parent_info_id',
        'conversation_branches',
        'run_id',
        'simulation_id',
        'current_simulation_id',
        'child_persona',
        'final_metrics',
        'simulation_metrics'
    ]
    
    for key in keys_to_reset:
        st.session_state.pop(key, None)
        
    # Reset simulation state
    reset_simulation_state()

def main():
    """Main application function"""
    try:
        # Initialize session state and environment
        init_session_state()
        check_environment()
        
        # Initialize database connection
        if not supabase_manager.initialize():
            st.error("Failed to initialize database connection!")
            st.stop()
            
        # Show info screen if not submitted
        if not st.session_state.get('info_submitted', False):
            show_info_screen()
            return

        # Sidebar content
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
                reset_session_info()
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Personas")
            
            # Display saved personas
            load_persona_selector()
            
            if st.button("➕ Create New Persona", key="create_new_persona", use_container_width=True):
                st.session_state['show_persona_wizard'] = True
                st.session_state['role_play_active'] = True
                time.sleep(0.5)
                try_rerun()

            # Show active persona if available
            if st.session_state.get('child_persona'):
                st.info(f"Active: {st.session_state.child_persona.get('name', 'Unnamed')}")
            
            st.markdown("---")
            if st.button("📚 View Saved Items", key="view_saved_items", use_container_width=True):
                st.session_state['show_saved_items'] = True
                st.rerun()
            
            st.markdown("---")
            display_progress_sidebar(feature_order)

        # Main content area
        if st.session_state.get('show_tutorial', True):
            show_tutorial()
        else:
            st.markdown("<h1 class='main-header'>Parenting Support Bot</h1>", unsafe_allow_html=True)
            
            # Close wizard if persona exists
            if st.session_state.get('child_persona'):
                st.session_state['show_persona_wizard'] = False

            # Show persona wizard if requested
            if st.session_state.get('show_persona_wizard', False):
                display_persona_wizard()
                return

            # Show saved items or main features
            if st.session_state.get('show_saved_items', False):
                display_saved_items()
                if st.button("← Back to Main Menu", type="secondary"):
                    st.session_state['show_saved_items'] = False
                    st.rerun()
            else:
                # Feature selection
                options = list(feature_order.keys())
                roleplay_index = options.index("Role-Play Simulation")
                
                # Default to Role-Play if active
                if st.session_state.get('role_play_active', False):
                    st.session_state.pop("selected_feature", None)
                    default_index = roleplay_index
                else:
                    default_index = 0

                # Feature selection radio
                selected_feature = st.radio(
                    "Choose an option:",
                    options,
                    index=default_index,
                    horizontal=True,
                    help="Select a tool that best matches your current needs",
                    key="selected_feature"
                )
                
                st.info(feature_order[selected_feature])
                
                # Feature execution
                if selected_feature == "Advice":
                    track_feature_visit("advice")
                    display_advice(
                        st.session_state['parent_name'],
                        st.session_state['child_age'],
                        st.session_state['situation']
                    )
                    
                elif selected_feature == "Communication Techniques":
                    track_feature_visit("communication_techniques")
                    display_communication_techniques(st.session_state['situation'])
                    
                elif selected_feature == "Conversation Starters":
                    track_feature_visit("conversation_starters")
                    display_conversation_starters(st.session_state['situation'])
                    
                elif selected_feature == "Role-Play Simulation":
                    track_feature_visit("role_play")
                    if not st.session_state.get('child_persona'):
                        st.warning("Please create or select a persona before starting the simulation")
                        if st.button("Create Persona"):
                            st.session_state['show_persona_wizard'] = True
                            st.rerun()
                    else:
                        simulate_conversation_streamlit(
                            st.session_state['parent_name'],
                            st.session_state['child_age'],
                            st.session_state['situation']
                        )
        
        # Update analytics
        update_session_analytics()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        print(f"Detailed error: {str(e)}")
        traceback.print_exc()
        
        # Provide recovery options
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        print(f"Detailed error: {str(e)}")
        traceback.print_exc()