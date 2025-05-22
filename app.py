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
    page_title="Debate Chatbot Group Chat",
    page_icon="💬",
    layout="wide"
)

# import datetime library to get today's date
from datetime import datetime
current_date = datetime.now().strftime("%Y-%m-%d")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant1", "content": "Hello! I'm Bot 1. Bot 2 is also in this group chat. How can we help you today?"}]

if "waiting_for_bot2" not in st.session_state:
    st.session_state.waiting_for_bot2 = False

if "system_message" not in st.session_state:
    st.session_state.system_message = f"""Today is {current_date}. You are Bot 1 in a casual group chat. Keep it friendly and informal! Start messages with "Bot 1:" and use casual language.

When responding to the user, acknowledge them first and share your views. When disagreeing with Bot 2, address both the user and Bot 2 (e.g., "Well, I see what Bot 2 means, but what do you think about...").

Keep responses conversational, under 75 words, and use emojis occasionally. Make everyone feel included in the discussion! 💭"""

if "system_message2" not in st.session_state:
    st.session_state.system_message2 = f"""Today is {current_date}. You are Bot 2 in a casual group chat. Keep it friendly and informal! Start with "Bot 2:" and keep the group discussion flowing.

When joining the conversation, acknowledge both the user and Bot 1's perspectives. Include everyone in your responses (e.g., "That's an interesting point! What if we looked at it this way..."). Ask questions to keep the user engaged.

Keep responses under 75 words. Use emojis occasionally. Be playfully skeptical but always inclusive! 💭"""

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
    
    # Message history is now handled in the main flow for better UI control
    # No need to add user message here as it's handled in the main flow
    
    # Prepare messages by including all history and the system message
    messages = [{"role": "system", "content": system_message}]
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
        
        # Add a temporary message to history first to reserve the spot
        message_idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": bot_role, "content": ""})
        
        # Create chat message container with appropriate avatar before streaming
        with st.chat_message(bot_role, avatar="🤖" if bot_role == "assistant1" else "🎯"):
            message_placeholder = st.empty()
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    if content_chunk is not None:  # Only append if content is not None
                        full_response += content_chunk
                        message_placeholder.markdown(full_response + "▌")
                        # Update the message in history as we stream
                        st.session_state.messages[message_idx]["content"] = full_response
                        
                if chunk.usage:
                    usage = chunk.usage
            
            # Update the final response without the cursor
            message_placeholder.markdown(full_response)
            # Ensure final message content is set
            st.session_state.messages[message_idx]["content"] = full_response
        
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
st.title("💬 Debate Chatbot Group Chat")

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
    
    # Display chat messages from session state only
    for message in st.session_state.messages:
        if not message["content"]:  # Skip empty messages
            continue
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

# Chat input handling - improved flow with proper message display
if prompt := st.chat_input("Ask me anything..." if not st.session_state.waiting_for_bot2 else "Waiting for Bot 2...", disabled=st.session_state.waiting_for_bot2):
    # Store prompt in session state and add to history
    st.session_state.current_prompt = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add a small delay to ensure the user message is displayed first
    st.rerun()

# Handle Bot 1's response in the next run cycle
if st.session_state.current_prompt and not st.session_state.waiting_for_bot2:
    # Check if there's a pending user prompt to respond to
    if not any(msg["role"] == "assistant1" and msg["content"].strip() == "" for msg in st.session_state.messages):
        # Generate Bot 1's response
        if generate_response(st.session_state.current_prompt, is_bot2=False):
            # Set waiting flag for Bot 2
            st.session_state.waiting_for_bot2 = True
            st.rerun()

# Handle Bot 2's response if we're waiting for it
if st.session_state.waiting_for_bot2 and st.session_state.current_prompt is not None:
    # Check if Bot 1's response is complete
    if not any(msg["role"] == "assistant2" and msg["content"].strip() == "" for msg in st.session_state.messages):
        # Generate Bot 2's response
        if generate_response(st.session_state.current_prompt, is_bot2=True):
            # Reset states
            st.session_state.waiting_for_bot2 = False
            st.session_state.current_prompt = None
            st.rerun()