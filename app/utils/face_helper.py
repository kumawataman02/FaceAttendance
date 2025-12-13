import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from app.services.face_recognition import face_service


def convert_numpy_types(obj):
    """Convert NumPy types to Python native types"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def find_best_match(
        query_embedding: List[float],
        students_with_embeddings: Dict[int, List[List[float]]],
        threshold: float = 0.6
) -> Tuple[Optional[int], float]:
    """
    Find the best matching student for a face embedding

    Returns: (student_id, similarity_score)
    """
    best_match_id = None
    best_similarity = 0.0

    for student_id, embeddings in students_with_embeddings.items():
        max_similarity = 0.0

        for student_emb in embeddings:
            similarity, _ = face_service.compare_faces(query_embedding, student_emb)
            max_similarity = max(max_similarity, similarity)

        if max_similarity > best_similarity and max_similarity >= threshold:
            best_similarity = max_similarity
            best_match_id = student_id

    return best_match_id, best_similarity