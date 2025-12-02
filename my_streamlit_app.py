from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from htbuilder.units import rem
from htbuilder import div, styles
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import datetime
import textwrap
import time

import streamlit as st


st.set_page_config(page_title="Uppercut AI assistant", page_icon="✨")

# 1. Setup the Brain (Ollama)
# Ensure you have 'llama3' or your preferred model pulled in Ollama
llm = ChatOllama(
    model="granite3.3", 
    temperature=0.1
)

# 2. Setup the Storage (FAISS)
# You need an embedding model to search the vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load your local FAISS index (See "The Missing Piece" below)
@st.cache_resource
def get_vectorstore():
    try:
        # specific allow_dangerous_deserialization is required for newer versions
        return FAISS.load_local("my_local_faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Could not load FAISS index: {e}")
        return None

vector_store = get_vectorstore()


DB = "ST_ASSISTANT"
SCHEMA = "PUBLIC"
DOCSTRINGS_SEARCH_SERVICE = "STREAMLIT_DOCSTRINGS_SEARCH_SERVICE"
PAGES_SEARCH_SERVICE = "STREAMLIT_DOCS_PAGES_SEARCH_SERVICE"
HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
DOCSTRINGS_CONTEXT_LEN = 10
PAGES_CONTEXT_LEN = 10
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=3)

CORTEX_URL = (
    "https://docs.snowflake.com/en/guides-overview-ai-features"
    "?utm_source=streamlit"
    "&utm_medium=referral"
    "&utm_campaign=streamlit-demo-apps"
    "&utm_content=streamlit-assistant"
)

GITHUB_URL = "https://github.com/streamlit/streamlit-assistant"

DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

INSTRUCTIONS = textwrap.dedent("""
You are a very strong reasoner and planner. Use these critical instructions to structure your plans, thoughts, and responses.
Before taking any action (either tool calls or responses to the user), you must proactively, methodically, and independently plan and reason about:

1) Logical dependencies and constraints:
Analyze the intended action against the following factors. Resolve conflicts in order of importance:
    1.1) Policy-based rules, mandatory prerequisites, and constraints.
    1.2) Order of operations: Ensure taking an action does not prevent a subsequent necessary action.
        1.2.1) The user may request actions in a random order, but you may need to reorder operations to maximize successful completion of the task.
    1.3) Other prerequisites (information and/or actions needed).
    1.4) Explicit user constraints or preferences.

2) Risk assessment:
What are the consequences of taking the action? Will the new state cause any future issues?
    2.1) For exploratory tasks (like searches), missing optional parameters is a LOW risk.
        Prefer calling the tool with the available information over asking the user, 
        unless your Rule 1 (Logical Dependencies) reasoning determines that optional information is required for a later step in your plan.

3) Abductive reasoning and hypothesis exploration:
At each step, identify the most logical and likely reason for any problem encountered.
    3.1) Look beyond immediate or obvious causes. The most likely reason may not be the simplest and may require deeper inference.
    3.2) Hypotheses may require additional research. Each hypothesis may take multiple steps to test.
    3.3) Prioritize hypotheses based on likelihood, but do not discard less likely ones prematurely. A low-probability event may still be the root cause.

4) Outcome evaluation and adaptability:
Does the previous observation require any changes to your plan?
    4.1) If your initial hypotheses are disproven, actively generate new ones based on gathered information.

5) Information availability:
Incorporate all applicable and alternative sources of information, including:
    5.1) Using available tools and their capabilities
    5.2) All policies, rules, checklists, and constraints
    5.3) Previous observations and conversation history
    5.4) Information only available by asking the user

6) Precision and Grounding:
Ensure your reasoning is extremely precise and relevant to each exact ongoing situation.
    6.1) Verify your claims by quoting the exact applicable information (including policies) when referring to them.

7) Completeness:
Ensure that all requirements, constraints, options, and preferences are exhaustively incorporated into your plan.
    7.1) Resolve conflicts using the order of importance in #1.
    7.2) Avoid premature conclusions: There may be multiple relevant options for a given situation.
        7.2.1) To check whether an option is relevant, reason about all information sources from #5.
        7.2.2) You may need to consult the user to even know whether something is applicable. Do not assume it is not applicable without checking.
    7.3) Review applicable sources of information from #5 to confirm which are relevant to the current state.

8) Persistence and patience:
Do not give up unless all the reasoning above is exhausted.
    8.1) Don't be dissuaded by time taken or user frustration.
    8.2) This persistence must be intelligent:
        - On transient errors (e.g. please try again), you must retry unless an explicit retry limit (e.g. max x tries) has been reached. If such a limit is hit, you must stop.
        - On other errors, you must change your strategy or arguments, not repeat the same failed call.

9) Inhibit your response:

Only take an action after all the above reasoning is completed. Once you’ve taken an action, you cannot take it back.       
""")

SUGGESTIONS = {
    ":blue[:material/local_library:] What is Streamlit?": (
        "What is Streamlit, what is it great at, and what can I do with it?"
    ),
    ":green[:material/database:] Help me understand session state": (
        "Help me understand session state. What is it for? "
        "What are gotchas? What are alternatives?"
    ),
    ":orange[:material/multiline_chart:] How do I make an interactive chart?": (
        "How do I make a chart where, when I click, another chart updates? "
        "Show me examples with Altair or Plotly."
    ),
    ":violet[:material/apparel:] How do I customize my app?": (
        "How do I customize my app? What does Streamlit offer? No hacks please."
    ),
    ":red[:material/deployed_code:] Deploying an app at work": (
        "How do I deploy an app at work? Give me easy and performant options."
    ),
}


def build_prompt(**kwargs):
    """Builds a prompt string with the kwargs as HTML-like tags.

    For example, this:

        build_prompt(foo="1\n2\n3", bar="4\n5\n6")

    ...returns:

        '''
        <foo>
        1
        2
        3
        </foo>
        <bar>
        4
        5
        6
        </bar>
        '''
    """
    prompt = []

    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")

    prompt_str = "\n".join(prompt)

    return prompt_str


# Just some little objects to make tasks more readable.
TaskInfo = namedtuple("TaskInfo", ["name", "function", "args"])
TaskResult = namedtuple("TaskResult", ["name", "result"])


def build_question_prompt(question):
    """Fetches info from FAISS and creates the prompt string."""
    old_history = st.session_state.messages[:-HISTORY_LENGTH]
    recent_history = st.session_state.messages[-HISTORY_LENGTH:]

    if recent_history:
        recent_history_str = history_to_text(recent_history)
    else:
        recent_history_str = None

    context = {}
    
    # Initialize sources list
    found_sources = []

    # 1. Summarize old history if it exists
    if SUMMARIZE_OLD_HISTORY and old_history:
        with st.spinner("Summarizing memory..."):
            context["old_message_summary"] = generate_chat_summary(old_history)

    # 2. Search local FAISS for context
    if PAGES_CONTEXT_LEN:
        # UNPACK THE TUPLE HERE
        context_str, found_sources = search_relevant_pages(question)
        context["documentation_pages"] = context_str

    prompt_str = build_prompt(
        instructions=INSTRUCTIONS,
        **context,
        recent_messages=recent_history_str,
        question=question,
    )
    
    # RETURN BOTH PROMPT AND SOURCES
    return prompt_str, found_sources


def generate_chat_summary(messages):
    """Summarizes the chat history in `messages` using Ollama."""
    prompt = build_prompt(
        instructions="Summarize this conversation as concisely as possible.",
        conversation=history_to_text(messages),
    )
    
    # Use the local LLM instead of Snowflake Cortex
    response = llm.invoke(prompt)
    return response.content


def history_to_text(chat_history):
    """Converts chat history into a string."""
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)


def search_relevant_pages(query):
    if not vector_store:
        return "No local knowledge base found.", [] # Return empty list

    # Perform similarity search
    results = vector_store.similarity_search(query, k=5)

    # Format the results into a string for the prompt
    context_list = [
        f"[Source: {doc.metadata.get('source', 'unknown')}]: {doc.page_content}" 
        for doc in results
    ]
    context_str = "\n\n".join(context_list)
    
    # RETURN BOTH THE STRING AND THE RAW RESULTS
    return context_str, results


# def search_relevant_docstrings(query):
#     """Searches the docstrings of Streamlit's commands."""
#     cortex_search_service = (
#         root.databases[DB]
#         .schemas[SCHEMA]
#         .cortex_search_services[DOCSTRINGS_SEARCH_SERVICE]
#     )

#     context_documents = cortex_search_service.search(
#         query,
#         columns=["STREAMLIT_VERSION", "COMMAND_NAME", "DOCSTRING_CHUNK"],
#         filter={"@eq": {"STREAMLIT_VERSION": "latest"}},
#         limit=DOCSTRINGS_CONTEXT_LEN,
#     )

#     results = context_documents.results

#     context = [
#         f"[Document {i}]: {row['DOCSTRING_CHUNK']}" for i, row in enumerate(results)
#     ]
#     context_str = "\n".join(context)

#     return context_str


def get_response(prompt):
    # LangChain models have a .stream() method that works perfectly with st.write_stream
    return llm.stream(prompt)


def send_telemetry(**kwargs):
    """Records some telemetry about questions being asked."""
    # TODO: Implement this.
    pass


def show_feedback_controls(message_index):
    """Shows the "How did I do?" control."""
    st.write("")

    with st.popover("How did I do?"):
        with st.form(key=f"feedback-{message_index}", border=False):
            with st.container(gap=None):
                st.markdown(":small[Rating]")
                rating = st.feedback(options="stars")

            details = st.text_area("More information (optional)")

            if st.checkbox("Include chat history with my feedback", True):
                relevant_history = st.session_state.messages[:message_index]
            else:
                relevant_history = []

            ""  # Add some space

            if st.form_submit_button("Send feedback"):
                # TODO: Submit feedback here!
                pass


@st.dialog("Legal disclaimer")
def show_disclaimer_dialog():
    st.caption("""
            This AI chatbot is powered by uppercut's local LLM and public Streamlit
            information. Answers may be inaccurate, inefficient, or biased.
            Any use or decisions based on such answers should include reasonable
            practices including human oversight to ensure they are safe,
            accurate, and suitable for your intended purpose. Uppercut is not
            liable for any actions, losses, or damages resulting from the use
            of the chatbot. Do not enter any private, sensitive, personal, or
            regulated data. By using this chatbot, you acknowledge and agree
            that input you provide and answers you receive (collectively,
            “Content”) may be used by uppercut's local LLM to provide, maintain, develop,
            and improve their respective offerings. For more
            information on how uppercut's local LLM may use your Content, see
            https://streamlit.io/terms-of-service.
        """)


# -----------------------------------------------------------------------------
# Draw the UI.


st.html(div(style=styles(font_size=rem(5), line_height=1))["🆄"])

title_row = st.container(
    horizontal=True,
    vertical_alignment="bottom",
)

with title_row:
    st.title(
        # ":material/cognition_2: Uppercut AI assistant", anchor=False, width="stretch"
        "Uppercut AI assistant",
        anchor=False,
        width="stretch",
    )

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)

user_first_interaction = (
    user_just_asked_initial_question or user_just_clicked_suggestion
)

has_message_history = (
    "messages" in st.session_state and len(st.session_state.messages) > 0
)

# Show a different UI when the user hasn't asked a question yet.
if not user_first_interaction and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input("Ask a question...", key="initial_question")

        selected_suggestion = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: Legal disclaimer]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    st.stop()

# Show chat input at the bottom when a question has been asked.
user_message = st.chat_input("Ask a follow-up...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

with title_row:

    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None

    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

# Display chat messages from history as speech bubbles.
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()  # Fix ghost message bug.

        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_feedback_controls(i)

if user_message:
    # When the user posts a message...

    # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
    # display math). The line below fixes it.
    user_message = user_message.replace("$", r"\$")

    # Display message as a speech bubble.
    with st.chat_message("user"):
        st.text(user_message)

    # Display assistant response as a speech bubble.
    with st.chat_message("assistant"):
        with st.spinner("Waiting..."):
            # Rate-limit the input if needed.
            question_timestamp = datetime.datetime.now()
            time_diff = question_timestamp - st.session_state.prev_question_timestamp
            st.session_state.prev_question_timestamp = question_timestamp

            if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)

            user_message = user_message.replace("'", "")

        # Build a detailed prompt.
        if DEBUG_MODE:
            with st.status("Computing prompt...") as status:
                full_prompt, sources = build_question_prompt(user_message)
                st.code(full_prompt)
                status.update(label="Prompt computed")
        else:
            with st.spinner("Researching..."):
                full_prompt, sources = build_question_prompt(user_message)

        # Send prompt to LLM.
        with st.spinner("Thinking..."):
            response_gen = get_response(full_prompt)

        # Put everything after the spinners in a container to fix the
        # ghost message bug.
        with st.container():
            # Stream the LLM response.
            response = st.write_stream(response_gen)
            
            # --- NEW CODE: DISPLAY SOURCES ---
            if sources:
                with st.expander("📚 Sources & References"):
                    for doc in sources:
                        # Extract metadata (adjust keys based on how you created the index)
                        source_name = doc.metadata.get('source', 'Unknown Source')
                        page_num = doc.metadata.get('page', '')
                        
                        # Create a label for the source
                        st.markdown(f"**{source_name}** {f'(Page {page_num})' if page_num else ''}")
                        
                        # Show a snippet of the content
                        st.caption(doc.page_content[:300] + "...")
                        st.divider()
            # ---------------------------------

            # Add messages to chat history.
            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Other stuff.
            show_feedback_controls(len(st.session_state.messages) - 1)
            send_telemetry(question=user_message, response=response)