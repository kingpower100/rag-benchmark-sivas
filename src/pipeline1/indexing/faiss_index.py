from pathlib import Path

import faiss
import numpy as np

from src.pipeline1.indexing.base import BaseVectorIndex


class FaissIndex(BaseVectorIndex):
    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self.index = None

    def build(self, embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim) if self.metric == "l2" else faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target))

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path)

    def search(self, query_embedding: np.ndarray, top_k: int):
        scores, indices = self.index.search(np.array([query_embedding], dtype="float32"), top_k)
        return scores[0], indices[0]

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    @property
    def dim(self) -> int | None:
        return int(self.index.d) if self.index is not None else None
