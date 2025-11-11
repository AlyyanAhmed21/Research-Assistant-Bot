
from typing import List, TypedDict, Dict, Optional

class GraphState(TypedDict):
    """
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
    """
    query: str
    preprocessed_query: str
    docs: List[str]
    candidates: Dict[str, str]
    critique: Dict
    best_candidate_id: Optional[str]
    final_answer: str
    retries: int
