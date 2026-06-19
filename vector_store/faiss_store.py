import faiss
import numpy as np
import pickle

class FAISSStore:
    def __init__(self, dim, index_path):
        self.index = faiss.IndexFlatIP(dim)   # 内积相似度（配合归一化向量等效余弦）
        self.id_to_doc = {}   # 内部ID -> Document
        self.index_path = index_path

    def add(self, vectors: np.ndarray, docs: List[Document]):
        start_id = self.index.ntotal
        self.index.add(vectors)
        for i, doc in enumerate(docs):
            self.id_to_doc[start_id + i] = doc

    def search(self, query_vec: np.ndarray, k: int):
        scores, indices = self.index.search(query_vec.reshape(1, -1), k)
        docs = [self.id_to_doc[i] for i in indices[0] if i != -1]
        return docs, scores[0][:len(docs)]

    def save(self):
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump(self.id_to_doc, f)

    def load(self):
        self.index = faiss.read_index(f"{self.index_path}.faiss")
        with open(f"{self.index_path}.pkl", "rb") as f:
            self.id_to_doc = pickle.load(f)