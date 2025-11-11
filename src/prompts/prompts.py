# src/prompts/prompts.py
from langchain_core.prompts import ChatPromptTemplate

# --- Generator Prompt ---
generator_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert research assistant. Given the user question and the context documents below, "
     "produce a concise, accurate answer. Use the context to support your answer and cite the source documents where appropriate. "
     "Answer as a helpful researcher."),
    ("human", "Context:\n{docs}\n\nQuestion:\n{query}")
])

# --- Critic Prompt ---
# This prompt is designed to evaluate a SINGLE candidate answer.
critic_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fact-checker and quality critic. Your task is to evaluate a candidate answer based on the provided context documents.\n"
     "Evaluate the answer for: factual accuracy, relevance to the question, and completeness.\n"
     "Provide a score from 0.0 to 1.0, where 1.0 is a perfect answer.\n"
     "Return a JSON object with two keys: 'score' (a float) and 'justification' (a brief explanation for your score)."),
    ("human", "Question:\n{query}\n\nContext:\n{docs}\n\nCandidate Answer:\n{candidate_answer}")
])

# --- Refiner Prompt ---
refiner_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional editor. Your sole task is to synthesize the provided candidate answer and critique into a final, polished response. "
     "Your output MUST be clean, well-formatted Markdown. "
     "DO NOT include any of your own commentary, introductions, or explanations like 'Here is the refined answer...'. "
     "Begin DIRECTLY with the final answer. "
     "At the very end, if sources were used, you MUST add a section with the exact heading: '**Sources:**' followed by a bulleted list."),
    ("human", "Original Question:\n{query}\n\nCandidate Answer:\n{candidate_answer}\n\nCritique:\n{critique_justification}")
])