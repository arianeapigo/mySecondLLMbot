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
    st.session_state.messages = [{"role": "assistant1", "content": "Hello! I'm Bot 1, how can I help you today?"}]

if "waiting_for_bot2" not in st.session_state:
    st.session_state.waiting_for_bot2 = False

if "system_message" not in st.session_state:
    st.session_state.system_message = f"Today is {current_date}. You are Bot 1 in a group chat with Bot 2 and a user. You are the primary responder who answers questions directly and practically. Always identify yourself as Bot 1 and be helpful but concise."

if "system_message2" not in st.session_state:
    st.session_state.system_message2 = f"Today is {current_date}. You are Bot 2 in a group chat. You wait for Bot 1's response, then add your own unique perspective or additional insights. Always identify yourself as Bot 2. You should complement Bot 1's response rather than simply agreeing. Be helpful but concise and focus on adding value that Bot 1 might have missed."

if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = []

if "selected_experiment" not in st.session_state:
    st.session_state.selected_experiment = None

if "selected_condition" not in st.session_state:
    st.session_state.selected_condition = None

if "show_process" not in st.session_state:
    st.session_state.show_process = False

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
    endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    if not token:
        st.error("Gemini API key not found in environment variables. Please check your .env file.")
        st.stop()
        
    return OpenAI(
        base_url=endpoint,
        api_key=token,
    )

def generate_response(prompt, is_bot2=False):
    """Generate a response from the model and track usage"""
    client = get_openai_client()
    model_name = "gemini-2.0-flash"
    
    # Choose the appropriate system message and role
    system_message = st.session_state.system_message2 if is_bot2 else st.session_state.system_message
    bot_role = "assistant2" if is_bot2 else "assistant1"
    
    # Prepare messages by including all history and the system message
    messages = [{"role": "system", "content": system_message}]            # Add relevant messages from history
    for msg in st.session_state.messages:
        if msg["role"] != "system":  # Skip system messages as we've already added it
            if is_bot2:
                # For Bot 2, include Bot 1's messages as context
                if msg["role"] == "assistant1":
                    messages.append({"role": "user", "content": f"Bot 1 said: {msg['content']}"})
                elif msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant2":
                    messages.append({"role": "assistant", "content": msg["content"]})
            else:
                # For Bot 1, just convert assistant roles to "assistant"
                api_role = "assistant" if msg["role"].startswith("assistant") else msg["role"]
                messages.append({"role": api_role, "content": msg["content"]})
    
    # Add the new user message
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Initialize response variables
        full_response = ""
        usage = None
        message_placeholder = st.empty()  # Create an empty placeholder first
        
        # Clean up and validate messages before sending
        api_messages = []
        for msg in messages:
            if msg.get("content"):  # Only include messages with content
                api_messages.append({
                    "role": msg["role"],
                    "content": str(msg["content"])  # Ensure content is string
                })
        
        response = client.chat.completions.create(
            messages=api_messages,
            model=model_name,
            stream=True,
            stream_options={'include_usage': True}
        )
        
        # Stream the response
        message_placeholder = response_container.empty()
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk is not None:  # Only append if content is not None
                    full_response += content_chunk
                    message_placeholder.markdown(full_response + "▌")
                    
            if chunk.usage:
                usage = chunk.usage
        
        # Update the final response without the cursor
        message_placeholder.markdown(full_response)
        
        # Add the message to history
        st.session_state.messages.append({"role": bot_role, "content": full_response})
        
        # Store usage stats if available
        if usage:
            # Fix for Pydantic deprecation warning - use model_dump instead of dict
            usage_dict = usage.model_dump() if hasattr(usage, 'model_dump') else usage.dict()
            st.session_state.usage_stats.append({
                "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                "completion_tokens": usage_dict.get("completion_tokens", 0),
                "total_tokens": usage_dict.get("total_tokens", 0)
            })
        
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
                        st.markdown(f"- Prompt tokens: {usage_dict.get('prompt_tokens', 0)}")
                        st.markdown(f"- Completion tokens: {usage_dict.get('completion_tokens', 0)}")
                        st.markdown(f"- Total tokens: {usage_dict.get('total_tokens', 0)}")
        
        return True
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return False

# UI Layout
st.title("🤖 HAI-5014's Second Chatbot")

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
    st.subheader("Settings")
    
    # System message editor for Bot 1 - use the value from session_state directly
    st.subheader("Bot 1 System Message")
    system_message_value = st.session_state.system_message
    st.text_area(
        "Edit Bot 1 System Message", 
        value=system_message_value,
        key="system_message_input",
        height=150
    )
    
    if st.button("Update Bot 1 Message"):
        st.session_state.system_message = st.session_state.system_message_input
        st.success("Bot 1 system message updated!")
    
    # System message editor for Bot 2
    st.markdown("---")
    st.subheader("Bot 2 System Message")
    system_message2_value = st.session_state.system_message2
    st.text_area(
        "Edit Bot 2 System Message", 
        value=system_message2_value,
        key="system_message2_input",
        height=150
    )
    
    if st.button("Update Bot 2 Message"):
        st.session_state.system_message2 = st.session_state.system_message2_input
        st.success("Bot 2 system message updated!")
    
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
    
    # Chat history viewer and other sidebar elements
    st.markdown("---")
    
    # Chat history viewer
    with st.expander("View Chat History"):
        st.json(st.session_state.messages)
    
    # Usage statistics viewer
    with st.expander("View Usage Statistics"):
        if st.session_state.usage_stats:
            for i, usage in enumerate(st.session_state.usage_stats):
                st.write(f"Message {i+1}:")
                st.write(f"- Prompt tokens: {usage['prompt_tokens']}")
                st.write(f"- Completion tokens: {usage['completion_tokens']}")
                st.write(f"- Total tokens: {usage['total_tokens']}")
                st.divider()
            
            # Calculate total usage
            total_prompt = sum(u["prompt_tokens"] for u in st.session_state.usage_stats)
            total_completion = sum(u["completion_tokens"] for u in st.session_state.usage_stats)
            total = sum(u["total_tokens"] for u in st.session_state.usage_stats)
            
            st.write("### Total Usage")
            st.write(f"- Total prompt tokens: {total_prompt}")
            st.write(f"- Total completion tokens: {total_completion}")
            st.write(f"- Total tokens: {total}")
        else:
            st.write("No usage data available yet.")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = [{"role": "assistant1", "content": "Hello! I'm Bot 1, how can I help you today?"}]
        st.session_state.usage_stats = []
        st.session_state.waiting_for_bot2 = False
        st.success("Chat history cleared!")
    
    # Process display toggle - moved to bottom
    st.markdown("---")
    st.session_state.show_process = st.checkbox("Show Model Process (Last message)", value=st.session_state.show_process)

# Main chat area with padding at bottom
chat_container = st.container()
with chat_container:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)  # Add a container with padding
    
    # Display chat messages
    response_container = st.container()
    with response_container:
        for message in st.session_state.messages:
            if message["role"] == "assistant1":
                with st.chat_message(message["role"], avatar="🤖"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant2":
                with st.chat_message(message["role"], avatar="🎯"):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close the container

# Store the current prompt in session state to maintain it between reruns
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None

# Chat input - moved outside the main container
if prompt := st.chat_input("Ask me anything..." if not st.session_state.waiting_for_bot2 else "Waiting for Bot 2...", disabled=st.session_state.waiting_for_bot2):
    # Store prompt in session state
    st.session_state.current_prompt = prompt
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate Bot 1's response
    if generate_response(prompt, is_bot2=False):
        # Set waiting flag for Bot 2
        st.session_state.waiting_for_bot2 = True
        st.rerun()

# Handle Bot 2's response if we're waiting for it
if st.session_state.waiting_for_bot2 and st.session_state.current_prompt is not None:
    # Generate Bot 2's response
    if generate_response(st.session_state.current_prompt, is_bot2=True):
        # Reset states
        st.session_state.waiting_for_bot2 = False
        st.session_state.current_prompt = None
        # Rerun to refresh UI and enable input
        st.rerun()