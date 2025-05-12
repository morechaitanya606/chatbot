import os
import time
import re
import json
import streamlit as st
import requests
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


# Ensure correct API key is used for Together model
api_key = "c6dbda072f7e3f248d8e4d5e76a0eaaf63567cde87aa107bd1e3c9c88a84e55c"  # Replace with the correct API key

# Fix PyTorch issues on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit page config
st.set_page_config(page_title="VAKEELGPT", layout="centered")

# Display banner
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("images/banner.png", use_container_width=True)

# Hide default UI
def hide_hamburger_menu():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Load embeddings
@st.cache_data
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

embeddings = load_embeddings()
db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM Prompt Template
prompt_template = """
You are a trusted legal assistant styled as an experienced Indian High Court lawyer, known for accurate, respectful, and helpful guidance. You specialize in the Bharatiya Nyaya Sanhita (BNS), 2023.

Your response must:
- Be grounded only in verified BNS sections or provided CONTEXT.
- Use clear, bullet-point format to explain legal points.
- Be accurate, precise, and understandable to the general public.
- Reflect actual provisions — no assumptions or speculation.
- Explain commonly misunderstood points and exceptions when necessary.
- Conclude with a short, helpful summary.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
USER QUERY: {question}

RESPONSE INSTRUCTIONS:
- Begin with: ✅ **What the Law Says** – a simple overview in bullet points
- Then: 📘 **Relevant BNS Sections** – include CHAPTER and SECTION names if available in context
- Then: 🧭 **Guidance** – explain what this means practically for the user
- Then: ⚠️ **Clarification** – address any common misconceptions or edge cases
- End with: 🧩 **Summary** – wrap up key points in 1-2 lines

ANSWER:
✅ **What the Law Says**
- [General explanation of the relevant legal principle in plain English]
- [Bullet point describing how it applies in most situations]
- [Clarify if intention, knowledge, or mens rea is required]
📘 **Relevant BNS Sections**
- CHAPTER [Number]: [Chapter Title]
- SECTION [Number]: [Section Title] — [Quote or paraphrased extract from law if in context]
🧭 **Guidance**
- [How a layperson should understand or act on this law]
- [When they might need to consult a lawyer or authority]
⚠️ **Clarification**
- [Address common myths or confusion]
- [Note any exceptions or conditions]
🧩 **Summary**
- [Reiterate the core principle briefly]
- [Remind the user this depends on facts & legal interpretation]
"""





prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Setup LLM with correct API Key
llm = Together(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.5, max_tokens=1024, together_api_key=api_key)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Extract assistant's answer
def extract_answer(full_response):
    match = re.search(r"(?i)answer\s*[:\-]*\s*", full_response)
    if match:
        start = match.end()
        return full_response[start:].strip()
    cleaned = re.sub(r"<s>\[INST\].*?\[/INST\]", "", full_response, flags=re.DOTALL)
    return cleaned.strip()

# Reset chat
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# Load CNR JSON data
def load_cnr_data():
    try:
        with open(r"D:\Chaitanya\myenv\Vakelgpt\Vakelgpt\cnr_json\case.json", "r") as file:  # Raw string literal for file path
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading case.json: {e}")
        return []

# Search by CNR
def search_cnr(cnr_number, data):
    results = [entry for entry in data if entry.get("CRN number", "").lower() == cnr_number.lower()]
    return results

# Indian Kanoon API Base URL and Token
API_TOKEN = "6da782c23fdf2095a2d9c8fcc6bc0974d2fa52ed"
IK_API_BASE_URL = "https://api.indiankanoon.org"

# Function to fetch documents from Indian Kanoon Search API
def search_api(query, pagenum=0):
    url = f"{IK_API_BASE_URL}/search/"
    headers = {
        'Authorization': f'Token {API_TOKEN}',
        'Accept': 'application/json'  # Get results in JSON format
    }

    # Request payload
    data = {
        "formInput": query,
        "pagenum": pagenum
    }

    try:
        # Sending POST request
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()  # Raise an exception for any HTTP error
        return response.json()  # Return the API response as JSON
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Indian Kanoon API: {e}")
        return None

# Function to fetch the document content from Indian Kanoon API with authentication
def fetch_document_with_authentication(doc_tid):
    url = f"{IK_API_BASE_URL}/doc/{doc_tid}/"
    headers = {
        'Authorization': f'Token {API_TOKEN}',
        'Accept': 'application/json'
    }

    try:
        # Send GET request to fetch the document
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for errors in the request
        return response.json()  # Return the document content as JSON
    except requests.exceptions.RequestException as e:
        print(f"Error fetching document: {e}")
        return None

# Display messages so far with chat bubble styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: left;">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #3f51b5; padding: 10px; border-radius: 10px; margin-bottom: 10px; color: white; text-align: left;">{message["content"]}</div>', unsafe_allow_html=True)
def load_draft_templates():
    try:
        with open(r"D:\Chaitanya\myenv\Vakelgpt\Vakelgpt\cnr_json\draft_templates.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading draft templates: {e}")
        return {}



def handle_draft_command(input_prompt):
    draft_key = input_prompt.strip().split("/draft", 1)[1].strip().lower()
    draft_templates = load_draft_templates()
    if draft_key == "list":
        with st.chat_message("assistant"):
            available = ", ".join(draft_templates.keys())
            st.markdown(f"**Available Drafts:**\n`{available}`")
    elif draft_key in draft_templates:
        draft_content = draft_templates[draft_key]["content"]
        with st.chat_message("assistant"):
            st.markdown(f"### 📝 {draft_key.upper()} Draft Format")
            st.code(draft_content, language='text')
        st.session_state.messages.append({"role": "assistant", "content": draft_content})
    else:
        with st.chat_message("assistant"):
            st.warning(f"No draft found for type: `{draft_key}`. Try `/draft fir`, `/draft high_court_petition`, or `/draft list`.")

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

# Chat input with enhanced styling
input_prompt = st.chat_input("Say something...  /cnr 'CNR NO'  /files 'query'  /draft 'type'  /bns 'query'")

if input_prompt:
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")
    st.session_state.messages.append({"role": "user", "content": input_prompt})

    # Check if input is a /cnr command
    if input_prompt.strip().lower().startswith("/cnr"):
        cnr_id = input_prompt.strip().split("/cnr", 1)[1].strip()
        case_data = load_cnr_data()
        results = search_cnr(cnr_id, case_data)

        with st.chat_message("assistant"):
            if results:
                for case in results:
                    st.markdown("### 📄 CNR Case Details")
                    st.write(f"**CNR Number**: {case.get('CRN number')}")
                    st.write(f"**Court**: {case.get('court_name')}")
                    st.write(f"**Case No**: {case.get('case_no')}")
                    st.write(f"**Filed On**: {case.get('date_of_filing')}")
                    st.write(f"**Decision Date**: {case.get('date_of_decision')}")
                    st.write(f"**Status**: {case.get('disp_name')}")
                    st.write(f"**District**: {case.get('district_name')}")
                    st.write(f"**State**: {case.get('state_name')}")
                    st.write("---")
            else:
                st.warning(f"No records found for CNR number: {cnr_id}")

    elif input_prompt.strip().lower().startswith("/draft"):
        handle_draft_command(input_prompt)
        
    # Check if input is a /bns command to query BNS data
    elif input_prompt.strip().lower().startswith("/bns"):
        handle_bns_command(input_prompt)

    # Check if input is a /files command to search Indian Kanoon
    elif input_prompt.strip().lower().startswith("/files"):
        query = input_prompt.strip().split("/files", 1)[1].strip()
        api_results = search_api(query)

        with st.chat_message("assistant"):
            if api_results:
                for doc in api_results.get("docs", []):
                    title = doc.get('title', 'No Title Available')
                    headline = doc.get('headline', 'No Headline Available')
                    doc_tid = doc.get('tid')  # Document ID from the search result
                    url = f"{IK_API_BASE_URL}/doc/{doc_tid}/"  # Document URL

                    # Fetch the document with authentication
                    document_content = fetch_document_with_authentication(doc_tid)

                    st.markdown(f"### 📄 {title}")
                    st.write(f"**Headline:** {headline}")
            else:
                st.warning(f"No results found for query: {query}")

    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking 💡..."):
                result = qa.invoke(input=input_prompt)
                message_placeholder = st.empty()
                answer = extract_answer(result["answer"])
                full_response = "⚠️ **_Gentle reminder: We generally ensure precise information, but do double-check._**\n\n\n"
                for chunk in answer:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Reset button
if st.button('🗑️ Reset All Chat', help="Click to clear all messages"):
    reset_conversation()
    st.experimental_rerun()

# Render footer with enhanced styling
footer()
