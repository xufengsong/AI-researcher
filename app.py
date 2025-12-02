import streamlit as st
import os
import shutil
import subprocess
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from htbuilder.units import rem
from htbuilder import div, styles
from collections import namedtuple
import datetime
import textwrap
import time

# --- Configuration ---
PROJECTS_ROOT = "projects"
INGESTION_SCRIPT_PATH = "ingestion_script.py"  # Ensure this is in the same folder or provide full path

st.set_page_config(page_title="Uppercut AI assistant", page_icon="✨", layout="wide")

# Ensure project root exists
os.makedirs(PROJECTS_ROOT, exist_ok=True)

# 1. Setup the Brain (Ollama)
llm = ChatOllama(
    model="granite3.3", 
    temperature=0.1
)

# 2. Setup the Storage (FAISS)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- Helper Functions for Project Management ---

def get_projects():
    """Returns a list of subdirectories in the projects root."""
    return [d for d in os.listdir(PROJECTS_ROOT) if os.path.isdir(os.path.join(PROJECTS_ROOT, d))]

def create_project(project_name):
    """Creates a new project directory."""
    path = os.path.join(PROJECTS_ROOT, project_name)
    if not os.path.exists(path):
        os.makedirs(path)
        # Create subfolders for raw files and the vector index
        os.makedirs(os.path.join(path, "files")) 
        os.makedirs(os.path.join(path, "index"))
        return True
    return False

def run_ingestion(project_name):
    """
    Triggers the external ingestion_script.py.
    Assumes the script takes arguments for source files and output path.
    """
    project_path = os.path.join(PROJECTS_ROOT, project_name)
    files_dir = os.path.join(project_path, "files")
    index_dir = os.path.join(project_path, "index")

    # Display a spinner while the subprocess runs
    with st.spinner(f"Ingesting documents for {project_name}..."):
        try:
            # MODIFY THIS COMMAND to match how your script accepts arguments
            cmd = [
                "python", INGESTION_SCRIPT_PATH,
                "--source", files_dir,
                "--output", index_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Ingestion complete!")
                # CLEAR CACHE to force reload of the FAISS index
                st.cache_resource.clear()
            else:
                st.error("Ingestion script failed.")
                st.code(result.stderr)
        except Exception as e:
            st.error(f"Error running ingestion script: {e}")

# Load local FAISS index based on SELECTED PROJECT
@st.cache_resource
def get_vectorstore(project_name):
    if not project_name:
        return None
        
    index_path = os.path.join(PROJECTS_ROOT, project_name, "index")
    
    # Check if index exists before trying to load
    if not os.path.exists(index_path) or not os.listdir(index_path):
        return None

    try:
        # specific allow_dangerous_deserialization is required for newer versions
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        # If the index folder exists but is corrupt or empty (handled above), return None
        return None

# --- SIDEBAR LOGIC ---
with st.sidebar:
    st.header("🗂️ Project Manager")
    
    # 1. Project Selector
    existing_projects = get_projects()
    
    # Session state for current project
    if "current_project" not in st.session_state:
        st.session_state.current_project = existing_projects[0] if existing_projects else None

    selected_project = st.selectbox(
        "Active Project", 
        existing_projects, 
        index=existing_projects.index(st.session_state.current_project) if st.session_state.current_project in existing_projects else 0
    )
    st.session_state.current_project = selected_project

    st.divider()

    # 2. Create New Project
    with st.expander("New Project"):
        new_proj_name = st.text_input("Project Name")
        if st.button("Create"):
            if new_proj_name:
                if create_project(new_proj_name):
                    st.success(f"Created {new_proj_name}")
                    st.rerun()
                else:
                    st.warning("Project already exists.")

    st.divider()

    # 3. File Upload & Ingestion
    if selected_project:
        st.subheader(f"Update: {selected_project}")
        uploaded_files = st.file_uploader(
            "Upload documents", 
            accept_multiple_files=True,
            type=['txt', 'pdf', 'md', 'csv']
        )
        
        if st.button("Process & Ingest"):
            if uploaded_files:
                # Save files to project directory
                save_path = os.path.join(PROJECTS_ROOT, selected_project, "files")
                
                for uploaded_file in uploaded_files:
                    with open(os.path.join(save_path, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Run the ingestion script
                run_ingestion(selected_project)
            else:
                st.warning("Please upload files first.")

# --- MAIN APP LOGIC ---

# Retrieve vector store for the SPECIFIC project
vector_store = get_vectorstore(st.session_state.current_project)

DB = "ST_ASSISTANT"
SCHEMA = "PUBLIC"
HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
PAGES_CONTEXT_LEN = 10
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=3)
DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

INSTRUCTIONS2 = textwrap.dedent("""
You are a very strong reasoner and planner. Use these critical instructions to structure your plans, thoughts, and responses.
Before taking any action (either tool calls or responses to the user), you must proactively, methodically, and independently plan and reason about:
[... truncated standard instructions to save space, keep your original instructions here ...]
""")

SUGGESTIONS = {
    "Explain this project": "Summarize the documents in this project.",
    "What is the key insight?": "What is the key insight from the provided context?",
}

def build_prompt(**kwargs):
    prompt = []
    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")
    return "\n".join(prompt)

def build_question_prompt(question):
    """Fetches info from FAISS and creates the prompt string."""
    old_history = st.session_state.messages[:-HISTORY_LENGTH]
    recent_history = st.session_state.messages[-HISTORY_LENGTH:]

    if recent_history:
        recent_history_str = history_to_text(recent_history)
    else:
        recent_history_str = None

    context = {}

    # 1. Summarize old history if it exists
    if SUMMARIZE_OLD_HISTORY and old_history:
        with st.spinner("Summarizing memory..."):
            context["old_message_summary"] = generate_chat_summary(old_history)

    # 2. Search local FAISS for context (ONLY IF VECTOR STORE EXISTS)
    if vector_store and PAGES_CONTEXT_LEN:
        context["documentation_pages"] = search_relevant_pages(question)
    elif not vector_store:
         context["system_note"] = "Note: No knowledge base (FAISS index) found for this project yet."

    return build_prompt(
        instructions=INSTRUCTIONS2,
        **context,
        recent_messages=recent_history_str,
        question=question,
    )

def generate_chat_summary(messages):
    prompt = build_prompt(
        instructions="Summarize this conversation as concisely as possible.",
        conversation=history_to_text(messages),
    )
    response = llm.invoke(prompt)
    return response.content

def history_to_text(chat_history):
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)

def search_relevant_pages(query):
    if not vector_store:
        return "No local knowledge base found."
    results = vector_store.similarity_search(query, k=5)
    context_list = [
        f"[Source: {doc.metadata.get('source', 'unknown')}]: {doc.page_content}" 
        for doc in results
    ]
    return "\n\n".join(context_list)

def get_response(prompt):
    return llm.stream(prompt)

def show_feedback_controls(message_index):
    # (Keep your existing feedback logic here)
    pass

def show_disclaimer_dialog():
    # (Keep your existing disclaimer logic here)
    pass


# -----------------------------------------------------------------------------
# Draw the UI.

st.html(div(style=styles(font_size=rem(5), line_height=1))["🆄"])

title_row = st.container(
    horizontal=True,
    vertical_alignment="bottom",
)

with title_row:
    # Update title to show current project
    proj_display = f" - {st.session_state.current_project}" if st.session_state.current_project else ""
    st.title(
        f"Uppercut AI{proj_display}",
        anchor=False,
        width="stretch",
    )

# Initialization Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear conversation logic
def clear_conversation():
    st.session_state.messages = []

# Chat Logic
if not st.session_state.messages:
    # Show empty state / suggestions
    pass

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()
        st.markdown(message["content"])

user_message = st.chat_input("Ask a question about this project...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_prompt = build_question_prompt(user_message)
            response_gen = get_response(full_prompt)
            
            with st.container():
                response = st.write_stream(response_gen)
                st.session_state.messages.append({"role": "assistant", "content": response})