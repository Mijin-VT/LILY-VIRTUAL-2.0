import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict

class RAGEngine:
    """Motor RAG (Retrieval-Augmented Generation) usando ChromaDB"""
    
    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.persist_dir = persist_dir
        
        # Inicializar cliente de Chroma
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Funcion de embedding (Sentence Transformers)
        # Usamos un modelo multilingüe ligero
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Obtener o crear colección
        self.collection = self.client.get_or_create_collection(
            name="lily_knowledge",
            embedding_function=self.embedding_fn
        )
        print(f"RAG Engine inicializado. Colección 'lily_knowledge' lista con {self.collection.count()} documentos.")

    def add_document(self, text: str, metadata: Dict = None, doc_id: str = None):
        """Agrega un documento a la base de datos vectorial"""
        if doc_id is None:
            import uuid
            doc_id = str(uuid.uuid4())
            
        if metadata is None:
            metadata = {}
            
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
    def add_conversation_turn(self, user_text: str, assistant_text: str):
        """Agrega un turno de conversación (Q&A) al contexto"""
        # Guardamos la interacción como un documento
        text = f"User: {user_text}\nLily: {assistant_text}"
        self.add_document(text, metadata={"type": "conversation", "timestamp": str(os.path.getmtime(__file__))}) # Timestamp simple placeholder

    def query(self, query_text: str, n_results: int = 3) -> List[str]:
        """Busca documentos relevantes para la consulta"""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            if results and results['documents']:
                return results['documents'][0]
            return []
        except Exception as e:
            print(f"Error en consulta RAG: {e}")
            return []
