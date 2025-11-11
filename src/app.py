# src/app.py
import gradio as gr
from dotenv import load_dotenv
import json
import re

from src.graph.graph import compile_graph

# Load environment variables
load_dotenv()

print("Compiling the research graph...")
app = compile_graph()
print("Graph compiled successfully!")

# --- ADVANCED CSS FOR A PROFESSIONAL AND APPEALING LOOK ---
CUSTOM_CSS = """
body { 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    background: #121212;   /* dark background for full page */
    color: #e0e0e0;        /* default text color */
    margin: 0; 
    padding: 0;
}

.gradio-container { 
    background: #121212; 
    padding: 20px 0; 
}

.main-column { 
    max-width: 800px; 
    margin: 0 auto; 
    padding: 0 15px;
}

.main-title { 
    text-align: center; 
    font-size: 2.2em; 
    font-weight: 700; 
    margin-bottom: 5px; 
    color: #4f8cff; 
}

.sub-title { 
    text-align: center; 
    margin-bottom: 20px; 
    font-size: 1em; 
    color: #b0b0b0; 
}

/* Input box and button */
textarea { 
    font-size: 1em; 
    padding: 12px; 
    border-radius: 8px; 
    background: #1e1e1e; 
    color: #f0f0f0; 
    border: 1px solid #333; 
    resize: vertical;
    min-height: 80px;
}

button { 
    font-size: 1em; 
    padding: 10px 18px; 
    border-radius: 8px; 
    background-color: #4f8cff; 
    color: white; 
    border: none; 
    cursor: pointer; 
    transition: background 0.2s;
}

button:hover { 
    background-color: #3670d6; 
}

/* Answer box */
.answer-box { 
    padding: 20px; 
    border-radius: 10px; 
    border: 1px solid #444; 
    background: #1e1e1e; 
    color: #f0f0f0; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); 
    font-size: 1em; 
    line-height: 1.5; 
    margin-top: 15px;
}

/* Sources box */
.sources-box { 
    margin-top: 15px; 
    padding: 15px; 
    border-radius: 10px; 
    background-color: #2a2a2a; 
    border: 1px solid #444; 
}
.sources-box ul { 
    margin: 0; 
    padding-left: 20px; 
    color: #b0b0b0; 
}
.sources-box li { margin-bottom: 6px; }
.sources-box p { 
    margin: 0 0 8px 0; 
    font-size: 1em; 
    font-weight: 600; 
    color: #e0e0e0; 
}
"""


def robust_parse_answer_and_sources(raw_answer: str) -> tuple[str, str]:
    """
    Robustly separates the raw LLM output into the main answer and a clean source list
    by looking for the exact '**Sources:**' heading.
    """
    # Use re.split to find the sources section. This is much more reliable.
    parts = re.split(r'\n\s*\*\*Sources:\*\*\s*\n', raw_answer, maxsplit=1, flags=re.IGNORECASE)
    
    answer = parts[0].strip()
    sources_html = "" # Default to empty if no sources are found

    if len(parts) > 1:
        sources_raw = parts[-1].strip()
        if sources_raw:
            sources_html = "<p><strong>Sources Used:</strong></p><ul>"
            # Split by newline and process each line as a potential list item
            for line in sources_raw.split('\n'):
                # Strip leading/trailing whitespace and remove list markers like '-', '*', 'o'
                clean_line = line.strip().lstrip('-*o ').strip()
                if clean_line:
                    sources_html += f"<li>{clean_line}</li>"
            sources_html += "</ul>"
            
    return answer, sources_html

def run_research(query: str):
    """
    Generator function for the Gradio interface with improved UX.
    """
    if not query:
        yield "Please enter a question.", "", "{}"
        return

    inputs = {"query": query}
    final_answer_text = ""
    
    # --- Streaming Logic ---
    # We will stream the raw output to the main answer box for a live feel.
    for event in app.stream(inputs):
        final_state = event
        if "refine" in event:
            partial_answer = event["refine"].get("final_answer", "")
            if partial_answer:
                final_answer_text += partial_answer
                # Yield the raw accumulating answer during stream
                yield final_answer_text, "...", "{}"

    # --- Final Post-Processing and Display ---
    # After the stream is complete, parse the full text for a clean, final display.
    clean_answer, sources_html = robust_parse_answer_and_sources(final_answer_text)
    final_state_json = json.dumps(final_state, indent=2, sort_keys=True)
    
    # The final yield will overwrite the raw stream with the perfectly parsed content.
    yield clean_answer, sources_html, final_state_json

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"), title="Research Assistant Bot", css=CUSTOM_CSS) as demo:
    with gr.Column(elem_classes=["main-column"]): # Center and constrain the layout
        gr.Markdown("# Research Assistant Bot", elem_classes=["main-title"])
        gr.Markdown("Enter your research question below. The bot intelligently uses local documents and web search, with every step validated by a multi-LLM critique pipeline.", elem_classes=["sub-title"])

        with gr.Row():
            query_input = gr.Textbox(label="Question", placeholder="e.g., What are the key features of the Llama-2 model?", lines=3, scale=3)
            submit_button = gr.Button("Run Research", variant="primary", scale=1)

        with gr.Tabs():
            with gr.TabItem("Final Answer"):
                # The answer box will display the clean, parsed answer
                output_answer = gr.Markdown(label="Synthesized Answer", value="Your answer will appear here...", elem_classes=["answer-box"])
                # The sources box will display the clean, formatted HTML list
                output_sources = gr.HTML(label="Sources & Citations", value="", elem_classes=["sources-box"])
            with gr.TabItem("Execution Trace"):
                output_state = gr.JSON(label="Final Graph State (For Debugging/Demo)")

    submit_button.click(
        fn=run_research,
        inputs=query_input,
        outputs=[output_answer, output_sources, output_state],
        api_name="run_research"
    )

if __name__ == "__main__":
    print("Launching Gradio UI...")
    demo.launch()