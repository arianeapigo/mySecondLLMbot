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
        # Container for the assistant's response in the chat interface
        response_container = st.chat_message("assistant")
        full_response = ""
        usage = None
        
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=True,
            stream_options={'include_usage': True}
        )
        
        # Stream the response
        message_placeholder = response_container.empty()
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                full_response += content_chunk
                message_placeholder.markdown(full_response + "▌")
                    
            if chunk.usage:
                usage = chunk.usage
        
        # Update the final response without the cursor
        message_placeholder.markdown(full_response)
        
        # Add the message to the appropriate history
        if bot_id == "bot1":
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.session_state.messages_bot2.append({"role": "assistant", "content": full_response})
        
        # Store usage stats if available
        if usage:
            # Fix for Pydantic deprecation warning - use model_dump instead of dict
            usage_dict = usage.model_dump() if hasattr(usage, 'model_dump') else usage.dict()
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
        
        # If show process is enabled, display the process details AFTER the response
        if st.session_state.show_process:
            process_container = st.container()
            with process_container:
                st.markdown("### Model Process")
                
                # Create expanders for process details - all collapsed by default
                request_expander = st.expander("Request Details", expanded=False)
                with request_expander:
                    st.markdown("**System Message:**")
                    st.code(system_message)
                    st.markdown("**User Input:**")
                    st.code(prompt)
                
                # Container for displaying raw response
                response_expander = st.expander("Raw Response", expanded=False)
                with response_expander:
                    st.code(full_response, language="markdown")
                
                # Container for usage stats
                if usage:
                    usage_expander = st.expander("Usage Statistics", expanded=False)
                    with usage_expander:
                        st.markdown("**Usage Statistics:**")
                        st.markdown(f"- Prompt tokens: {usage_dict.get("prompt_tokens", 0)}")
                        st.markdown(f"- Completion tokens: {usage_dict.get("completion_tokens", 0)}")
                        st.markdown(f"- Total tokens: {usage_dict.get("total_tokens", 0)}")
        
        return True
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return False

# UI Layout
st.title("🤖 HAI-5014's Dual Chatbot")

# Add CSS to make the input box stick to the bottom
st.markdown("""
    <style>
    .stChatFloatingInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        padding: 1rem !important;
        width: calc(100% - 250px) !important; /* Adjust for sidebar width */
        background-color: white !important;
        z-index: 1000 !important;
    }
    .main-content {
        padding-bottom: 100px; /* Add space at the bottom for the fixed input */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.subheader("Chatbot Selection")
    
    # Add radio buttons to switch between chatbots
    selected_bot = st.radio(
        "Select Active Chatbot",
        ["Chatbot 1", "Chatbot 2"],
        index=0 if st.session_state.active_bot == "bot1" else 1,
        key="bot_selector"
    )
    
    # Update active bot based on selection
    st.session_state.active_bot = "bot1" if selected_bot == "Chatbot 1" else "bot2"
    
    st.subheader("Settings")
    
    # Determine which system message to display based on active bot
    if st.session_state.active_bot == "bot1":
        system_message_value = st.session_state.system_message
        system_message_key = "system_message_input"
    else:
        system_message_value = st.session_state.system_message_bot2
        system_message_key = "system_message_bot2_input"
    
    # System message editor
    st.text_area(
        f"Edit System Message for {selected_bot}", 
        value=system_message_value,
        key=system_message_key,
        height=150
    )
    
    if st.button("Update System Message"):
        if st.session_state.active_bot == "bot1":
            st.session_state.system_message = st.session_state.system_message_input
        else:
            st.session_state.system_message_bot2 = st.session_state.system_message_bot2_input
        st.success(f"System message updated for {selected_bot}!")
    
    # Experiment loader section
    st.markdown("---")
    st.subheader("Experiment Loader")
    
    experiments = load_experiments()
    if experiments:
        # First dropdown: select experiment with a blank default option
        experiment_names = ["Select an experiment"] + [exp.get('experiment_name', "Unnamed Experiment") for exp in experiments]
        exp_index = st.selectbox(
            "Select Experiment", 
            range(len(experiment_names)),
            format_func=lambda i: experiment_names[i]
        ) 
        
        # Only proceed if a valid experiment is selected (not the blank option)
        if exp_index > 0:
            selected_experiment = experiments[exp_index - 1]  # Adjust index for the actual experiment
            
            # Second dropdown: select condition within the experiment with a blank default
            if 'conditions' in selected_experiment and selected_experiment['conditions']:
                condition_names = ["Select a condition"] + [cond.get('label', f"Condition {i+1}") 
                                  for i, cond in enumerate(selected_experiment['conditions'])]
                cond_index = st.selectbox(
                    "Select Condition", 
                    range(len(condition_names)),
                    format_func=lambda i: condition_names[i]
                )
                
                # Only enable the load button if a valid condition is selected
                if cond_index > 0:
                    selected_condition = selected_experiment['conditions'][cond_index - 1]  # Adjust index
                    
                    # Preview the system message
                    st.markdown("### Preview: System Message")
                    system_prompt = selected_condition.get('system_prompt', "You are a helpful assistant.")
                    # Fix empty label warning by providing a label
                    st.text_area(
                        "System Prompt Preview", 
                        value=system_prompt, 
                        height=120, 
                        disabled=True, 
                        key="preview_system_message",
                        label_visibility="collapsed"  # Hide the label but still provide one
                    )
                    
                    # Load button
                    if st.button("Load Experiment"):
                        # Update system message
                        st.session_state.system_message = system_prompt
                        
                        # Clear chat and start with opening message
                        opening_message = selected_condition.get('opening_message', "How can I help you today?")
                        st.session_state.messages = [{"role": "assistant", "content": opening_message}]
                        st.session_state.usage_stats = []
                        
                        # Save selected experiment and condition
                        st.session_state.selected_experiment = experiment_names[exp_index]
                        st.session_state.selected_condition = condition_names[cond_index]
                        
                        st.success(f"Loaded: {experiment_names[exp_index]} - {condition_names[cond_index]}")
                        st.rerun()
            else:
                st.warning("Selected experiment has no conditions.")
    else:
        st.warning("No experiment files found in the 'prompts' directory.")
    
    # Chat history viewer
    with st.expander("View Chat History"):
        if st.session_state.active_bot == "bot1":
            st.json(st.session_state.messages)
        else:
            st.json(st.session_state.messages_bot2)
    
    # Usage statistics viewer
    with st.expander("View Usage Statistics"):
        usage_stats = st.session_state.usage_stats if st.session_state.active_bot == "bot1" else st.session_state.usage_stats_bot2
        
        if usage_stats:
            for i, usage in enumerate(usage_stats):
                st.write(f"Message {i+1}:")
                st.write(f"- Prompt tokens: {usage['prompt_tokens']}")
                st.write(f"- Completion tokens: {usage['completion_tokens']}")
                st.write(f"- Total tokens: {usage['total_tokens']}")
                st.divider()
            
            # Calculate total usage
            total_prompt = sum(u["prompt_tokens"] for u in usage_stats)
            total_completion = sum(u["completion_tokens"] for u in usage_stats)
            total = sum(u["total_tokens"] for u in usage_stats)
            
            st.write("### Total Usage")
            st.write(f"- Total prompt tokens: {total_prompt}")
            st.write(f"- Total completion tokens: {total_completion}")
            st.write(f"- Total tokens: {total}")
        else:
            st.write("No usage data available yet.")
    
    # Clear chat button
    if st.button(f"Clear {selected_bot} Chat"):
        if st.session_state.active_bot == "bot1":
            st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
            st.session_state.usage_stats = []
        else:
            st.session_state.messages_bot2 = [{"role": "assistant", "content": "I'm the second assistant. How can I help?"}]
            st.session_state.usage_stats_bot2 = []
        st.success(f"{selected_bot} chat history cleared!")
    
    st.markdown("---")
    st.session_state.show_process = st.checkbox("Show Model Process (Last message)", value=st.session_state.show_process)

# Determine which messages to display based on the active bot
active_messages = st.session_state.messages if st.session_state.active_bot == "bot1" else st.session_state.messages_bot2
active_system_message = st.session_state.system_message if st.session_state.active_bot == "bot1" else st.session_state.system_message_bot2

# Main chat area with padding at bottom
chat_container = st.container()
with chat_container:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)  # Add a container with padding
    
    # Display currently active bot label
    st.markdown(f"### Currently Active: **{selected_bot}**")
    
    # Display chat messages for the active bot
    for message in active_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close the container

# Chat input - moved outside the main container
if prompt := st.chat_input(f"Ask {selected_bot} anything..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to appropriate history
    if st.session_state.active_bot == "bot1":
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Generate and display response for bot1
        generate_response(prompt, st.session_state.system_message, "bot1")
    else:
        st.session_state.messages_bot2.append({"role": "user", "content": prompt})
        # Generate and display response for bot2
        generate_response(prompt, st.session_state.system_message_bot2, "bot2")