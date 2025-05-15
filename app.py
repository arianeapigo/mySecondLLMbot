import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import json
import glob

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="HAI-5014's Second Chatbot",
    page_icon="🤖",
    layout="wide"
)

# import datetime library to get today's date
from datetime import datetime
current_date = datetime.now().strftime("%Y-%m-%d")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

if "messages_bot2" not in st.session_state:
    st.session_state.messages_bot2 = [{"role": "assistant", "content": "I'm the second assistant. How can I help?"}]

if "system_message" not in st.session_state:
    st.session_state.system_message = f"Today is {current_date}. You are a helpful assistant."

if "system_message_bot2" not in st.session_state:
    st.session_state.system_message_bot2 = f"Today is {current_date}. You are a creative assistant focused on innovation."

if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = []

if "usage_stats_bot2" not in st.session_state:
    st.session_state.usage_stats_bot2 = []

if "selected_experiment" not in st.session_state:
    st.session_state.selected_experiment = None

if "selected_condition" not in st.session_state:
    st.session_state.selected_condition = None

if "show_process" not in st.session_state:
    st.session_state.show_process = False

if "active_bot" not in st.session_state:
    st.session_state.active_bot = "bot1"

def load_experiments():
    """Load all experiment JSON files from the prompts directory"""
    experiments = []
    prompts_dir = os.path.join(os.getcwd(), "prompts")
    
    if not os.path.exists(prompts_dir):
        return experiments
    
    json_files = glob.glob(os.path.join(prompts_dir, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                experiments.append(data)
        except Exception as e:
            st.warning(f"Error loading {file_path}: {str(e)}")
    
    return experiments

def get_openai_client():
    """Create and return an OpenAI client configured with environment variables"""
    token = os.getenv("GEMINI_KEY")
    # Use os.getenv to get the endpoint instead of hardcoding it
    endpoint = os.getenv("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/openai/")
    
    if not token:
        st.error("Gemini API key not found in environment variables. Please check your .env file.")
        st.stop()
        
    return OpenAI(
        base_url=endpoint,
        api_key=token,
    )

def generate_response(prompt, system_message, bot_id="bot1"):
    """Generate a response from the model and track usage"""
    client = get_openai_client()
    model_name = os.getenv("LLM_MODEL")
    
    # Determine which message and usage history to use
    if bot_id == "bot1":
        messages_history = st.session_state.messages
        usage_stats = st.session_state.usage_stats
    else:
        messages_history = st.session_state.messages_bot2
        usage_stats = st.session_state.usage_stats_bot2
    
    # Prepare messages by including all history and the system message
    messages = [{"role": "system", "content": system_message}]
    
    # Add all previous messages from history
    for msg in messages_history:
        if msg["role"] != "system":  # Skip system messages as we've already added it
            messages.append(msg)
    
    # Add the new user message
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Create a response without streaming for the side-by-side view
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=False
        )
        
        # Get the full response
        full_response = response.choices[0].message.content
        
        # Add the message to the appropriate history
        if bot_id == "bot1":
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.session_state.messages_bot2.append({"role": "assistant", "content": full_response})
        
        # Store usage stats if available
        usage_dict = None
        if hasattr(response, 'usage') and response.usage:
            usage_dict = response.usage.model_dump() if hasattr(response.usage, 'model_dump') else response.usage.dict()
            stats_entry = {
                "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                "completion_tokens": usage_dict.get("completion_tokens", 0),
                "total_tokens": usage_dict.get("total_tokens", 0)
            }
            
            # Store in the appropriate usage stats
            if bot_id == "bot1":
                st.session_state.usage_stats.append(stats_entry)
            else:
                st.session_state.usage_stats_bot2.append(stats_entry)
        
        # Show process details if enabled
        if st.session_state.show_process:
            # Create a container for process details (will appear below the chatbots)
            process_container = st.container()
            
            with process_container:
                st.markdown(f"### Process Details - {bot_id.capitalize()}")
                
                col_sys, col_resp = st.columns(2)
                
                with col_sys:
                    with st.expander("System Message", expanded=False):
                        st.code(system_message)
                    
                    with st.expander("Full Conversation", expanded=False):
                        for msg in messages:
                            st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")
                
                with col_resp:
                    with st.expander("Response", expanded=False):
                        st.code(full_response)
                    
                    if usage_dict:
                        with st.expander("Usage Stats", expanded=False):
                            st.write(f"Prompt tokens: {usage_dict.get('prompt_tokens', 0)}")
                            st.write(f"Completion tokens: {usage_dict.get('completion_tokens', 0)}")
                            st.write(f"Total tokens: {usage_dict.get('total_tokens', 0)}")
        
        return True
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return False

# UI Layout
st.title("🤖 HAI-5014's Dual Chatbot")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Tabs for bot settings
    tab1, tab2 = st.tabs(["Chatbot 1", "Chatbot 2"])
    
    with tab1:
        # System message editor for Bot 1
        st.subheader("System Message")
        system_message_bot1 = st.text_area(
            "Edit System Message", 
            value=st.session_state.system_message,
            key="system_message_input",
            height=150,
            label_visibility="collapsed"
        )
        
        if st.button("Update System Message", key="update_system_1"):
            st.session_state.system_message = system_message_bot1
            st.success("System message updated for Chatbot 1!")
        
        if st.button("Clear Chat History", key="clear_chat_1"):
            st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
            st.session_state.usage_stats = []
            st.success("Chat history cleared for Chatbot 1")
    
    with tab2:
        # System message editor for Bot 2
        st.subheader("System Message")
        system_message_bot2 = st.text_area(
            "Edit System Message", 
            value=st.session_state.system_message_bot2,
            key="system_message_bot2_input",
            height=150,
            label_visibility="collapsed"
        )
        
        if st.button("Update System Message", key="update_system_2"):
            st.session_state.system_message_bot2 = system_message_bot2
            st.success("System message updated for Chatbot 2!")
        
        if st.button("Clear Chat History", key="clear_chat_2"):
            st.session_state.messages_bot2 = [{"role": "assistant", "content": "I'm the second assistant. How can I help?"}]
            st.session_state.usage_stats_bot2 = []
            st.success("Chat history cleared for Chatbot 2")
    
    # Process toggle
    st.markdown("---")
    show_process = st.checkbox("Show Model Process", value=st.session_state.show_process, key="process_toggle")
    if show_process != st.session_state.show_process:
        st.session_state.show_process = show_process
        st.rerun()
    
    # Experiments section if needed
    if st.session_state.selected_experiment:
        st.markdown("---")
        st.subheader("Current Experiment")
        st.write(f"**Experiment:** {st.session_state.selected_experiment['name']}")
        if st.session_state.selected_condition:
            st.write(f"**Condition:** {st.session_state.selected_condition['name']}")
    
    # Chat history and usage statistics
    st.markdown("---")
    
    with st.expander("Chatbot 1 Usage Stats", expanded=False):
        if st.session_state.usage_stats:
            total_prompt = sum(u["prompt_tokens"] for u in st.session_state.usage_stats)
            total_completion = sum(u["completion_tokens"] for u in st.session_state.usage_stats)
            total_tokens = sum(u["total_tokens"] for u in st.session_state.usage_stats)
            
            st.metric("Total Tokens", total_tokens)
            st.metric("Prompt Tokens", total_prompt)
            st.metric("Completion Tokens", total_completion)
        else:
            st.info("No usage data available")
    
    with st.expander("Chatbot 2 Usage Stats", expanded=False):
        if st.session_state.usage_stats_bot2:
            total_prompt = sum(u["prompt_tokens"] for u in st.session_state.usage_stats_bot2)
            total_completion = sum(u["completion_tokens"] for u in st.session_state.usage_stats_bot2)
            total_tokens = sum(u["total_tokens"] for u in st.session_state.usage_stats_bot2)
            
            st.metric("Total Tokens", total_tokens)
            st.metric("Prompt Tokens", total_prompt)
            st.metric("Completion Tokens", total_completion)
        else:
            st.info("No usage data available")

# Use columns for the side-by-side layout
col1, col2 = st.columns(2)

# First Chatbot Column
with col1:
    st.header("Chatbot 1")
    
    # Container for chat messages with fixed height
    chat_container1 = st.container()
    
    with chat_container1:
        # Create styled container with scrolling
        st.markdown("""
            <style>
            .chat-container {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
            }
            </style>
            <div class="chat-container" id="chat1">
        """, unsafe_allow_html=True)
        
        # Display messages for bot 1 using standard markdown
        # (st.chat_message doesn't work well in scrollable containers)
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                        <div style="background-color: #E9F5FD; padding: 10px; border-radius: 10px; max-width: 80%;">
                            <strong>You:</strong> {message['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="display: flex; margin-bottom: 10px;">
                        <div style="background-color: #F0F2F6; padding: 10px; border-radius: 10px; max-width: 80%;">
                            <strong>Assistant:</strong> {message['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Input for Chatbot 1
    prompt_bot1 = st.text_input("Ask Chatbot 1", key="input_bot1")
    if st.button("Send", key="send_bot1") or prompt_bot1:
        if prompt_bot1:  # Only proceed if there's text input
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt_bot1})
            
            # Generate response for bot1
            generate_response(prompt_bot1, st.session_state.system_message, "bot1")
            
            # Clear the input field
            st.session_state.input_bot1 = ""
            
            # Rerun to refresh the UI
            st.rerun()

# Second Chatbot Column
with col2:
    st.header("Chatbot 2")
    
    # Container for chat messages with fixed height
    chat_container2 = st.container()
    
    with chat_container2:
        # Create styled container with scrolling
        st.markdown("""
            <div class="chat-container" id="chat2">
        """, unsafe_allow_html=True)
        
        # Display messages for bot 2
        for message in st.session_state.messages_bot2:
            if message["role"] == "user":
                st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                        <div style="background-color: #E9F5FD; padding: 10px; border-radius: 10px; max-width: 80%;">
                            <strong>You:</strong> {message['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="display: flex; margin-bottom: 10px;">
                        <div style="background-color: #F0F2F6; padding: 10px; border-radius: 10px; max-width: 80%;">
                            <strong>Assistant:</strong> {message['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Input for Chatbot 2
    prompt_bot2 = st.text_input("Ask Chatbot 2", key="input_bot2")
    if st.button("Send", key="send_bot2") or prompt_bot2:
        if prompt_bot2:  # Only proceed if there's text input
            # Add user message to history
            st.session_state.messages_bot2.append({"role": "user", "content": prompt_bot2})
            
            # Generate response for bot2
            generate_response(prompt_bot2, st.session_state.system_message_bot2, "bot2")
            
            # Clear the input field
            st.session_state.input_bot2 = ""
            
            # Rerun to refresh the UI
            st.rerun()