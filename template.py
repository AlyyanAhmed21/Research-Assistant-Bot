# template.py
import os
import textwrap

# --- Project Structure Definition ---
# Define the directory and file structure of the project.
# Directories are ended with a '/'.
project_structure = [
    ".github/workflows/",
    "configs/.env.example",
    "data/corpus/sample_doc.txt",
    "notebooks/01_exploration.ipynb",
    "src/__init__.py",
    "src/app.py",
    "src/core/__init__.py",
    "src/core/parsers.py",
    "src/graph/__init__.py",
    "src/graph/graph.py",
    "src/graph/nodes.py",
    "src/graph/state.py",
    "src/prompts/__init__.py",
    "src/prompts/prompts.py",
    "src/retriever/__init__.py",
    "src/retriever/retriever.py",
    ".gitignore",
    "README.md",
    "requirements.txt",
]

# --- Boilerplate Content for Files ---
# Using a dictionary to map file paths to their initial content.
file_content = {
    "README.md": textwrap.dedent("""
        # Research Assistant Bot (LangChain + LangGraph + Multi-LLM RAG)

        This project implements a sophisticated Research Assistant Bot using a modern stack. It's designed to be run locally and scaled as needed.

        ## Architecture
        ```
        +--------+      +---------+       +-------------------+      +------------------+
        |  User  | -->  |  API /  |  -->  |   LangGraph Flow   | -->  |   Output / UI    |
        | (CLI / |      | Frontend|       | (Node orchestration)|      | (Gradio / UI)    |
        | Gradio)|      +---------+       +--------+-----------+      +------------------+
                                                  |
                                                  v
                       +----------------- LangChain (LLMChains) -----------------+
                       |  +----------+   +----------+   +----------+   +--------+ |
                       |  | Retriever|   | Generator|   | Critic   |   | Refiner| |
                       |  +----+-----+   +----+-----+   +----+-----+   +----+---+ |
                       |       |              |              |              |     |
                       |       v              v              v              v     |
                       |    Vector DB      Mistral-7B     Gemma/LLama    Mistral    |
                       +---------------------------------------------------------+
                                                  |
                                                  v
                                             LangSmith (traces, prompts)
        ```

        ## Features
        - **Multi-LLM RAG**: Uses multiple LLMs for generation, critique, and refinement.
        - **LangGraph Orchestration**: A robust, stateful graph manages the flow of logic.
        - **Local First**: Designed to run with local models (Ollama) and vector stores (Chroma).
        - **Observable**: Integrated with LangSmith for easy tracing and debugging.
        - **Modular**: Code is organized into logical components (retriever, graph, prompts).

        ## Setup & Installation

        1.  **Clone the repository:**
            ```bash
            git clone <your-repo-url>
            cd langgraph-research-assistant
            ```

        2.  **Create Conda Environment:**
            ```bash
            conda create --name research-bot python=3.11 -y
            conda activate research-bot
            ```

        3.  **Install Dependencies:**
            ```bash
            pip install -r requirements.txt
            ```

        4.  **Configure Environment Variables:**
            Copy the example and fill in your details.
            ```bash
            cp configs/.env.example .env
            ```
            Then edit `.env` with your API keys (especially for LangSmith).

        5.  **Run the application:**
            ```bash
            python src/app.py
            ```
    """),

    ".gitignore": textwrap.dedent("""
        # Python
        __pycache__/
        *.pyc
        *.pyo
        *.pyd
        .Python
        build/
        develop-eggs/
        dist/
        downloads/
        eggs/
        .eggs/
        lib/
        lib64/
        parts/
        sdist/
        var/
        wheels/
        *.egg-info/
        .installed.cfg
        *.egg
        .ipynb_checkpoints

        # Environment
        .env
        .venv
        env/
        venv/
        ENV/

        # Vector Store
        /chroma_db/
    """),

    "requirements.txt": textwrap.dedent("""
        # Core
        langchain
        langgraph
        langchain-community
        langchain-core
        langsmith

        # LLMs (local)
        langchain-ollama
        ollama

        # Vector Store & Embeddings
        chromadb
        sentence-transformers

        # Frontend/API
        gradio
        fastapi
        uvicorn

        # Utilities
        python-dotenv
        rich # for pretty printing
    """),

    "configs/.env.example": textwrap.dedent("""
        # LangSmith Configuration
        LANGCHAIN_TRACING_V2="true"
        LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY"
        LANGCHAIN_PROJECT="research-assistant-bot"

        # Ollama Configuration
        # OLLAMA_BASE_URL="http://localhost:11434" # Default, uncomment if needed

        # Model Names
        GENERATOR_MODEL="mistral"
        CRITIC_MODEL="gemma:2b"
        REFINER_MODEL="mistral"
    """),

    "data/corpus/sample_doc.txt": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It is built on top of LangChain and is designed for creating complex, cyclical graphs that are common in agentic architectures.",

    "src/app.py": textwrap.dedent("""
        import gradio as gr
        from dotenv import load_dotenv
        # from src.graph.graph import compile_graph # To be implemented

        # Load environment variables. This is important for LangSmith tracing.
        load_dotenv()

        print("Environment variables loaded.")
        # print("Compiling the research graph...")
        # app = compile_graph() # Your compiled LangGraph
        print("Graph compiled successfully.")

        def run_research(query: str):
            \"\"\"Function to be called by the Gradio interface.\"\"\"
            if not query:
                return "Please enter a question."

            # This is where you'll invoke your LangGraph
            # For now, it's a placeholder
            # inputs = {"query": query}
            # final_state = app.invoke(inputs)
            # return final_state.get("final_answer", "No answer found.")

            # --- Placeholder Logic ---
            print(f"Received query: {query}")
            return f"This is a placeholder response for the query: '{query}'. The graph is not yet implemented."


        # --- Gradio Interface ---
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Research Assistant Bot")
            gr.Markdown("Enter your research question below and get a synthesized answer with sources.")

            with gr.Row():
                query_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g., What is LangGraph and how does it work?",
                    lines=2
                )
                submit_button = gr.Button("Run Research")

            with gr.Column():
                output_answer = gr.Markdown(label="Answer")

            submit_button.click(
                fn=run_research,
                inputs=query_input,
                outputs=output_answer
            )

        if __name__ == "__main__":
            print("Launching Gradio UI...")
            demo.launch()
    """),

    "src/graph/state.py": textwrap.dedent("""
        from typing import List, TypedDict, Dict, Optional

        class GraphState(TypedDict):
            \"\"\"
            Represents the state of our graph.

            Attributes:
                query: The initial user query.
                preprocessed_query: The cleaned/normalized query.
                docs: A list of retrieved documents.
                candidates: A dictionary of generated answers from different models.
                critique: Scores and comments from the critic model.
                best_candidate_id: The ID of the best-scoring candidate.
                final_answer: The final, refined answer.
                retries: The number of times the generation step has been retried.
            \"\"\"
            query: str
            preprocessed_query: str
            docs: List[str]
            candidates: Dict[str, str]
            critique: Dict
            best_candidate_id: Optional[str]
            final_answer: str
            retries: int
    """),

    "src/graph/nodes.py": textwrap.dedent("""
        from src.graph.state import GraphState

        # These are placeholder stubs. We will implement them in the next steps.

        def preprocess_node(state: GraphState) -> GraphState:
            print("--- NODE: PREPROCESS ---")
            # Logic to preprocess the query
            state['preprocessed_query'] = state['query'].strip().lower()
            state['retries'] = 0
            return state

        def retrieve_node(state: GraphState) -> GraphState:
            print("--- NODE: RETRIEVE ---")
            # Logic to retrieve documents from vector store
            state['docs'] = ["Placeholder doc 1", "Placeholder doc 2"]
            return state

        def generate_node(state: GraphState) -> GraphState:
            print("--- NODE: GENERATE ---")
            # Logic to generate answers from multiple LLMs
            state['candidates'] = {
                "A": "Generated answer from Model A",
                "B": "Generated answer from Model B"
            }
            return state

        def critic_node(state: GraphState) -> GraphState:
            print("--- NODE: CRITIC ---")
            # Logic to score and critique the generated answers
            state['critique'] = {
                "scores": {"A": 0.9, "B": 0.6},
                "best_id": "A"
            }
            state['best_candidate_id'] = "A"
            return state

        def decision_node(state: GraphState):
            print("--- NODE: DECISION ---")
            # Logic to decide the next step based on critique
            # This will return a string: 'refine' or 'retry'
            if state['critique']['scores'][state['best_candidate_id']] > 0.7:
                return "refine"
            else:
                return "retry"

        def refine_node(state: GraphState) -> GraphState:
            print("--- NODE: REFINE ---")
            # Logic to refine the best answer
            best_answer = state['candidates'][state['best_candidate_id']]
            state['final_answer'] = f"Refined Answer: {best_answer}"
            return state

        def retry_node(state: GraphState) -> GraphState:
            print("--- NODE: RETRY ---")
            # Logic to handle retries, maybe by changing prompts or params
            state['retries'] += 1
            if state['retries'] > 2:
                # If we've retried too many times, just refine the best we have
                return "refine"
            else:
                return "generate"
    """),

    "src/graph/graph.py": textwrap.dedent("""
        from langgraph.graph import StateGraph, END
        from src.graph.state import GraphState
        from src.graph.nodes import (
            preprocess_node,
            retrieve_node,
            generate_node,
            critic_node,
            decision_node,
            refine_node,
            retry_node,
        )

        def create_graph():
            \"\"\"Creates the LangGraph workflow.\"\"\"
            workflow = StateGraph(GraphState)

            # Add nodes
            workflow.add_node("preprocess", preprocess_node)
            workflow.add_node("retrieve", retrieve_node)
            workflow.add_node("generate", generate_node)
            workflow.add_node("critic", critic_node)
            workflow.add_node("refine", refine_node)

            # Set entry and edges
            workflow.set_entry_point("preprocess")
            workflow.add_edge("preprocess", "retrieve")
            workflow.add_edge("retrieve", "generate")
            workflow.add_edge("generate", "critic")

            # Conditional edges from the decision node
            workflow.add_conditional_edges(
                "critic",
                decision_node,
                {
                    "refine": "refine",
                    "retry": "generate" # Simple retry, loops back to generate
                }
            )

            workflow.add_edge("refine", END)

            return workflow

        def compile_graph():
            \"\"\"Compiles the graph into a runnable app.\"\"\"
            app = create_graph().compile()
            return app

        if __name__ == "__main__":
            # A simple test to visualize the graph
            app = compile_graph()
            try:
                # You need to have graphviz installed for this to work
                # pip install pygraphviz
                img_data = app.get_graph().draw_png()
                with open("graph.png", "wb") as f:
                    f.write(img_data)
                print("Graph visualization saved to graph.png")
            except ImportError:
                print("Could not generate graph visualization. Please install pygraphviz: `pip install pygraphviz`")
    """),

    "src/prompts/prompts.py": textwrap.dedent("""
        from langchain_core.prompts import ChatPromptTemplate

        # --- Generator Prompt ---
        generator_prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert research assistant. Given the user question and the context documents below, "
             "produce a concise, accurate answer and include 2-3 short citations (source id and short snippet). "
             "Answer as a helpful researcher."),
            ("human", "Context:\n{docs}\n\nQuestion:\n{query}")
        ])

        # --- Critic Prompt ---
        critic_prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a fact-checker and quality critic. Given the question, context documents, and a candidate answer, "
             "evaluate the answer for factual accuracy, relevance, and completeness. "
             "Return a JSON object with two keys: 'score' (a float from 0.0 to 1.0) and 'justification' (a brief explanation)."),
            ("human", "Question:\n{query}\n\nContext:\n{docs}\n\nCandidate Answer:\n{candidate_answer}")
        ])

        # --- Refiner Prompt ---
        refiner_prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior researcher and synthesizer. Use the best candidate answer and the critic's commentary "
             "to produce a final, polished answer that is accurate, concise, and includes citations. "
             "Cite sources inline as [source_id]."),
            ("human", "Best Candidate Answer:\n{best_answer}\n\nCritique:\n{critique}")
        ])
    """),

    "src/retriever/retriever.py": textwrap.dedent("""
        # This file will contain the logic for setting up ChromaDB,
        # loading documents, creating embeddings, and building the retriever.
        # We will implement this in the next steps.

        def get_retriever():
            \"\"\"
            Placeholder for creating a vector store retriever.
            \"\"\"
            print("Retriever not implemented yet.")
            return None
    """),
}

# --- Script to Create the Project Structure ---
def create_project():
    """Generates the project structure and files."""
    print("Initializing project: Research Assistant Bot")

    for path in project_structure:
        if path.endswith('/'):
            # This is a directory
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        else:
            # This is a file
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            if not os.path.exists(path):
                with open(path, 'w', encoding='utf-8') as f:
                    content = file_content.get(path)
                    if content:
                        f.write(content)
                print(f"Created file: {path}")
            else:
                print(f"File already exists: {path}")

    print("\nProject structure created successfully!")
    print("Next steps:")
    print("1. Create and activate your Conda environment.")
    print("2. Run 'pip install -r requirements.txt'.")
    print("3. Copy 'configs/.env.example' to '.env' and add your API keys.")

if __name__ == "__main__":
    create_project()