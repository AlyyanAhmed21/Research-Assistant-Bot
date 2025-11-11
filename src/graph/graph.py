# src/graph/graph.py
from langgraph.graph import StateGraph, END
from src.graph.state import GraphState
from src.graph.nodes import (
    preprocess_node,
    retrieve_node,
    generate_node,
    critic_node,
    decision_node,
    refine_node,
    # --- Import the new nodes ---
    web_search_decision_node,
    web_search_node
)

def create_graph():
    """Creates the LangGraph workflow with a web search fallback."""
    workflow = StateGraph(GraphState)

    # --- Add Nodes ---
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refine", refine_node)

    # --- Set Entry and Edges ---
    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        web_search_decision_node,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "critic")

    workflow.add_conditional_edges(
        "critic",
        decision_node,
        {
            "refine": "refine",
            "generate": "generate",
        }
    )

    workflow.add_edge("refine", END)

    return workflow

# --- ADDING THIS FUNCTION BACK ---
def compile_graph():
    """Compiles the graph into a runnable LangChain object."""
    app = create_graph().compile()
    return app

# --- ADDING THIS BLOCK BACK ---
if __name__ == "__main__":
    # This block allows us to visualize the graph structure if needed.
    print("Compiling graph and generating visualization...")
    app = compile_graph()
    try:
        img_data = app.get_graph().draw_png()
        with open("graph_with_web_search.png", "wb") as f:
            f.write(img_data)
        print("Graph visualization saved to graph_with_web_search.png")
    except ImportError as e:
        print(f"Could not generate graph visualization: {e}")
        print("Please install pygraphviz and the graphviz system library.")