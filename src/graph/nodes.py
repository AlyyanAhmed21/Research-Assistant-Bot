import os
from typing import List

from dotenv import load_dotenv
from rich.console import Console
# ... (other imports) ...
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain_mistralai import ChatMistralAI

from src.graph.state import GraphState
from src.prompts.prompts import (
    generator_prompt_template,
    critic_prompt_template,
    refiner_prompt_template,
)
from src.retriever.retriever import create_retriever

load_dotenv()

# --- NEW DEBUG LINE ---
print(f"--- DEBUG: Loading model named '{os.getenv('GENERATOR_MODEL')}' ---")

# --- Initialize Rich Console for better printing ---
console = Console()

# Change the Tavily import and initialization

web_search_tool = TavilySearch(k=3)

console.print("[bold green]Initialization complete![/bold green]")

# --- Constants & Configuration ---
MAX_RETRIES = 2
SCORE_THRESHOLD = 0.7

# --- LLM and Retriever Initialization ---
# We initialize these components once to be reused by the nodes.
# This is more efficient than creating them in each node call.
try:
    console.print("[bold cyan]Initializing models and retriever...[/bold cyan]")

    # Initialize the retriever
    retriever = create_retriever(k_results=3)

    # NEW, CORRECT CODE using Mistral AI
    generator_llm = ChatMistralAI(
        model_name=os.getenv("GENERATOR_MODEL"),
        temperature=0.7,
    )

    critic_llm = ChatMistralAI(
        model_name=os.getenv("CRITIC_MODEL"),
        temperature=0.0, # Critic should be deterministic
    )

    refiner_llm = ChatMistralAI(
        model_name=os.getenv("REFINER_MODEL"),
        temperature=0.5,
    )
    console.print("[bold green]Initialization complete![/bold green]")
except Exception as e:
    console.print(f"[bold red]Error during initialization: {e}[/bold red]")
    # Exit or handle the error as appropriate
    exit()


# --- Pydantic Model for Critic Output ---
class Critique(BaseModel):
    score: float = Field(description="The score from 0.0 to 1.0 evaluating the answer.", ge=0, le=1)
    justification: str = Field(description="A brief justification for the score.")

# --- Node Implementations ---

def preprocess_node(state: GraphState) -> GraphState:
    console.print("\n--- NODE: PREPROCESS ---")
    query = state['query'].strip()
    state['preprocessed_query'] = query
    state['retries'] = 0
    state['candidates'] = {} # Ensure candidates dict is initialized
    console.log(f"Preprocessed query: '{query}'")
    return state

def retrieve_node(state: GraphState) -> GraphState:
    console.print("\n--- NODE: RETRIEVE ---")
    query = state['preprocessed_query']
    retrieved_docs: List[Document] = retriever.invoke(query)
    # Format docs into a string for the context
    docs_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    state['docs'] = docs_text
    console.log(f"Retrieved {len(retrieved_docs)} document chunks.")
    return state

def generate_node(state: GraphState) -> GraphState:
    console.print("\n--- NODE: GENERATE ---")
    generation_chain = generator_prompt_template | generator_llm | StrOutputParser()
    
    # For now, we generate one candidate. This can be extended to parallel generation.
    console.log("Generating candidate answer...")
    candidate_answer = generation_chain.invoke({
        "query": state['preprocessed_query'],
        "docs": state['docs']
    })
    
    state['candidates']['A'] = candidate_answer
    console.log("Candidate 'A' generated.")
    # console.print(f"[yellow]Candidate Answer:\n{candidate_answer}[/yellow]")
    return state

def critic_node(state: GraphState) -> GraphState:
    console.print("\n--- NODE: CRITIC ---")
    critic_chain = critic_prompt_template | critic_llm | JsonOutputParser(pydantic_object=Critique)
    
    critiques = {}
    best_score = -1.0
    best_candidate_id = None

    for cid, answer in state['candidates'].items():
        console.log(f"Critiquing candidate '{cid}'...")
        try:
            critique_result: Critique = critic_chain.invoke({
                "query": state['preprocessed_query'],
                "docs": state['docs'],
                "candidate_answer": answer
            })
            critiques[cid] = critique_result
            score = critique_result.get('score', 0.0) # Use .get() for safety
            justification = critique_result.get('justification', 'No justification provided.')
            console.log(f"Critique for '{cid}': Score={score}, Justification='{justification}'")

            if score > best_score:
                best_score = score
                best_candidate_id = cid
        except Exception as e:
            console.print(f"[bold red]Error critiquing candidate {cid}: {e}[/bold red]")
            # Assign a default low score if parsing fails
            critiques[cid] = {"score": 0.0, "justification": f"Failed to parse critique: {e}"}

    state['critique'] = critiques
    state['best_candidate_id'] = best_candidate_id
    console.log(f"Best candidate identified: '{best_candidate_id}' with score {best_score}")
    return state

def decision_node(state: GraphState) -> str:
    console.print("\n--- NODE: DECISION ---")
    best_cid = state['best_candidate_id']
    if best_cid is None:
        console.log("No valid candidates. Forcing a retry by looping to generate.")
        return "generate"# If something went wrong in critic, retry

    best_score = state['critique'][best_cid]['score']
    
    if best_score >= SCORE_THRESHOLD:
        console.log(f"Decision: Score ({best_score}) is above threshold ({SCORE_THRESHOLD}). Proceeding to refine.")
        return "refine"
    else:
        if state['retries'] < MAX_RETRIES:
            console.log(f"Decision: Score ({best_score}) is below threshold. Retrying (Attempt {state['retries'] + 1}/{MAX_RETRIES}).")
            state['retries'] += 1
            return "generate" # Loop back to generate
        else:
            console.log(f"Decision: Max retries reached. Proceeding with best available answer despite low score.")
            return "refine" # Fallback to refining the best we have

def refine_node(state: GraphState) -> GraphState:
    console.print("\n--- NODE: REFINE ---")
    refiner_chain = refiner_prompt_template | refiner_llm | StrOutputParser()
    
    best_cid = state['best_candidate_id']
    best_answer = state['candidates'][best_cid]
    critique_justification = state['critique'][best_cid]['justification']
    
    console.log("Refining the best answer...")
    final_answer = refiner_chain.invoke({
        "query": state['preprocessed_query'],
        "candidate_answer": best_answer,
        "critique_justification": critique_justification
    })
    
    state['final_answer'] = final_answer
    console.log("Answer refined successfully.")
    # console.print(f"[bold green]Final Answer:\n{final_answer}[/bold green]")
    return state

# --- New Nodes for Web Search ---

def web_search_decision_node(state: GraphState) -> str:
    """
    Determines whether to proceed with generation or to perform a web search.

    Returns:
        str: "generate" if local documents are found, "web_search" otherwise.
    """
    console.print("\n--- NODE: WEB SEARCH DECISION ---")
    if state['docs'] and len(state['docs']) > 10: # Check if docs string is not empty/trivial
        console.log("Decision: Local documents found. Proceeding to generation.")
        return "generate"
    else:
        console.log("Decision: No relevant local documents found. Proceeding to web search.")
        return "web_search"

def web_search_node(state: GraphState) -> GraphState:
    """
    Performs a web search using the Tavily API.

    Returns:
        GraphState: The state updated with the web search results in the 'docs' field.
    """
    console.print("\n--- NODE: WEB SEARCH ---")
    query = state['preprocessed_query']
    console.log(f"Searching web for: '{query}'")
    
    search_results = web_search_tool.invoke({"query": query})
    
    # Format the search results into a string for the context
    if search_results:
        docs_text = "\n\n---\n\n".join(search_results)
        state['docs'] = docs_text
        console.log("Web search successful.")
    else:
        state['docs'] = "No relevant information found on the web."
        console.log("Web search did not return any results.")
        
    return state