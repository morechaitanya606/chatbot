import os
import time
import re
import json
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together
from footer import footer

# Import BNS integration
try:
    from bns_integration import generate_bns_response
    BNS_AVAILABLE = True
    print("BNS integration loaded successfully")
except Exception as e:
    BNS_AVAILABLE = False
    print(f"Error loading BNS integration: {e}")

# API key for Together API
together_api_key = "c6dbda072f7e3f248d8e4d5e76a0eaaf63567cde87aa107bd1e3c9c88a84e55c"  # Together API key

# Available models
MODELS = {
    "mixtral": {
        "name": "Mixtral-8x22B",
        "id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "provider": "together"
    },
    "qwen": {
        "name": "Qwen3-235B",
        "id": "Qwen/Qwen3-235B-A22B-fp8-tput",
        "provider": "together"
    }
}

# Default model
DEFAULT_MODEL = "mixtral"

# Fix PyTorch issues on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit page config with improved styling
st.set_page_config(
    page_title="VAKEELGPT - AI Legal Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state variables
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    model_name = MODELS[DEFAULT_MODEL]["name"]
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"🤖 **Welcome to VakeelGPT!**\n\nI'm powered by {model_name} and ready to assist with your legal questions about Indian law, particularly the Bharatiya Nyaya Sanhita (BNS). How can I help you today?"
    })

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)

# Custom CSS for better chat UI
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
        border-left: 5px solid #4f46e5;
    }
    .chat-message.assistant {
        background-color: #4f46e5;
        color: white;
        border-left: 5px solid #818cf8;
    }
    .chat-header {
        font-size: 0.85rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .chat-bubble {
        padding: 0.5rem 1rem;
        border-radius: 1rem;
    }
    .stButton button {
        background-color: #4f46e5;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #3730a3;
        transform: translateY(-2px);
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: #4f46e5;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1.2rem;
    }
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Display header
st.markdown("""
<div class="main-header">
    <h1>⚖️ VakeelGPT</h1>
    <p>Your AI Legal Assistant for Indian Law</p>
</div>
""", unsafe_allow_html=True)

# Model selector in sidebar
with st.sidebar:
    st.title("Model Settings")

    # Model selection
    model_options = {key: info["name"] for key, info in MODELS.items()}
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.selected_model),
        help="Choose which AI model to use for legal assistance"
    )

    # Update session state if model changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.memory.clear()  # Clear memory when switching models

    st.markdown("---")
    st.markdown("### About the Models")

    if selected_model == "mixtral":
        st.markdown("""
        **Mixtral-8x22B** is a powerful mixture-of-experts model that excels at:
        - Complex legal reasoning
        - Detailed explanations
        - Understanding nuanced legal questions
        """)
    elif selected_model == "qwen":
        st.markdown("""
        **Qwen3-235B** is Alibaba's advanced large language model that offers:
        - Strong legal reasoning capabilities
        - Excellent performance on complex tasks
        - Comprehensive understanding of legal contexts
        """)

    st.markdown("---")
    st.markdown("### Tips for Better Results")
    st.markdown("""
    - Be specific in your legal questions
    - Mention relevant laws or sections if you know them
    - Use /bns command for Bharatiya Nyaya Sanhita queries
    """)

    # Expand sidebar by default
    st.sidebar.markdown('<script>document.querySelector("[data-testid=\'stSidebar\']").click();</script>', unsafe_allow_html=True)

# Load embeddings
@st.cache_data
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

# Load vector database
embeddings = load_embeddings()
db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Enhanced LLM Prompt Template for better legal responses
prompt_template = """
You are a precise legal assistant specializing in the Bharatiya Nyaya Sanhita (BNS), India's new criminal code replacing the Indian Penal Code.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
USER QUERY: {question}

CRITICAL INSTRUCTIONS:
1. ONLY provide information that is EXPLICITLY mentioned in the BNS text or the provided context
2. NEVER cite section numbers unless you can verify them in the context
3. ALWAYS include the chapter number and section title when citing a section
4. VERIFY that any section you cite actually relates to the topic being discussed
5. DO NOT exceed 100 words in your response

FOR CONCEPTS NOT EXPLICITLY DEFINED IN BNS (like "digital arrest"):
1. Begin with: "The term '[term]' is not explicitly defined in the Bharatiya Nyaya Sanhita."
2. If appropriate, briefly explain what the term commonly refers to (in 1-2 sentences)
3. DO NOT suggest specific BNS sections unless you can verify they exist AND are relevant
4. If mentioning potentially relevant sections, use language like:
   "This concept may relate to offenses described in Chapter [X] of BNS, though the term itself is not used."
5. NEVER fabricate or guess at section numbers, penalties, or legal definitions

RESPONSE STRUCTURE:
- Direct factual statement about what BNS actually contains (or doesn't contain)
- Only verified section citations with chapter numbers and section titles
- Clear distinction between what is explicitly in BNS and what is interpretation
- Acknowledgment of limitations when information is not available

ANSWER:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Function to get the appropriate LLM based on selected model
def get_llm(model_key):
    model_info = MODELS[model_key]

    # All models now use Together API
    return Together(
        model=model_info["id"],
        temperature=0.1,  # Lower temperature for more factual responses
        max_tokens=1024,
        together_api_key=together_api_key
    )

# Create function to get the conversational chain with the selected model
def get_qa_chain(model_key):
    llm = get_llm(model_key)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

# Extract assistant's answer from model response
def extract_answer(full_response):
    match = re.search(r"(?i)answer\s*[:\-]*\s*", full_response)
    if match:
        start = match.end()
        return full_response[start:].strip()
    cleaned = re.sub(r"<s>\[INST\].*?\[/INST\]", "", full_response, flags=re.DOTALL)
    return cleaned.strip()

# Reset chat function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

    # Add a system message showing which model is being used
    model_name = MODELS[st.session_state.selected_model]["name"]
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"🤖 **VakeelGPT initialized with {model_name}**\n\nI'm ready to assist with your legal questions about Indian law, particularly the Bharatiya Nyaya Sanhita (BNS). How can I help you today?"
    })

# Handle BNS query command
def handle_bns_command(input_prompt):
    query = input_prompt.strip().split("/bns", 1)[1].strip()

    with st.chat_message("assistant"):
        with st.spinner("Searching BNS data..."):
            if BNS_AVAILABLE:
                response = generate_bns_response(query)

                if response["success"]:
                    st.markdown(f"### 📚 Bharatiya Nyaya Sanhita Information")
                    st.markdown(response["answer"])

                    if response["sources"]:
                        st.markdown("### Sources")
                        for source in response["sources"]:
                            st.markdown(f"- {source}")
                else:
                    st.warning("Sorry, I couldn't find specific information about that in the BNS. Please try a different query.")
            else:
                st.error("BNS data integration is not available. Please check the server logs for details.")

        # Add the response to the chat history
        if BNS_AVAILABLE:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"] if response["success"] else "Sorry, I couldn't find specific information about that in the BNS."
            })

# Display chat messages with improved styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat input with enhanced styling
input_prompt = st.chat_input("Ask me about Indian law or use /bns for specific BNS queries...")

if input_prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(input_prompt)

    # Add to message history
    st.session_state.messages.append({"role": "user", "content": input_prompt})

    # Check if input is a /bns command to query BNS data
    if input_prompt.strip().lower().startswith("/bns"):
        handle_bns_command(input_prompt)
    else:
        # Process regular chat with the model
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {MODELS[st.session_state.selected_model]['name']} 💡..."):
                # Get the QA chain with the selected model
                qa_chain = get_qa_chain(st.session_state.selected_model)

                # Get response from the model
                result = qa_chain.invoke(input=input_prompt)
                message_placeholder = st.empty()
                answer = extract_answer(result["answer"])

                # Add disclaimer and model info
                model_name = MODELS[st.session_state.selected_model]["name"]
                full_response = f"⚠️ **_Note: This information is based on the Bharatiya Nyaya Sanhita text. For legal advice, please consult a qualified legal professional._**\n\n"

                # Stream the response
                for chunk in answer:
                    full_response += chunk
                    time.sleep(0.01)  # Slightly faster typing effect
                    message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

                # Final display without cursor
                message_placeholder.markdown(full_response, unsafe_allow_html=True)

            # Add to message history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Reset button with improved styling
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    model_name = MODELS[st.session_state.selected_model]["name"]
    if st.button(f'🗑️ Reset Conversation ({model_name})', help=f"Click to clear all messages and restart with {model_name}"):
        reset_conversation()
        st.rerun()

# Render footer with enhanced styling
footer()
