
import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Compute text embeddings using local sentence-transformers model. """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Convert a list of texts into a 2D numpy array of embeddings."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings= True,
            show_progress_bar=True
        )

        return embeddings