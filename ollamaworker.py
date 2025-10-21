import os
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import requests

class OllamaDocumentQA:
    def __init__(self, ollama_base_url: str = "http://localhost:11434", embedding_model: str = "embeddinggemma"):
        """
        Initialize the Ollama Document QA system.
        
        Args:
            ollama_base_url (str): Base URL for Ollama API
            embedding_model (str): Model to use for embeddings
        """
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.document_embeddings = {}
        self.file_contents = {}
        self.directory_listing = ""
        
    def set_directory(self, directory_path: str):
        self.directory_listing = qa_system.get_directory_listing_string(directory_path)

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Ollama.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embed",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1024  # Adjust size based on your model
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1 (List[float]): First vector
            vec2 (List[float]): Second vector
            
        Returns:
            float: Cosine similarity score
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def get_directory_listing_string(self, directory_path: str = ".") -> str:
        """
        Returns a string with directories and files in tree-like format.
        """
        output_string = ""
        
        try:
            path = Path(directory_path)
            
            if not path.exists():
                return f"Error: Directory '{directory_path}' does not exist."
            if not path.is_dir():
                return f"Error: '{directory_path}' is not a directory."
            
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            directories = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            for directory in directories:
                output_string += f"{directory.name}\n"
                try:
                    dir_files = sorted([f.name for f in directory.iterdir() if f.is_file()])
                    for file in dir_files:
                        output_string += f"    {file}\n"
                except PermissionError:
                    output_string += "    [Permission denied]\n"
                output_string += "\n"
            
            if files:
                output_string += ".\n"
                for file in sorted([f.name for f in files]):
                    output_string += f"    {file}\n"
                    
        except Exception as e:
            output_string = f"Error: {e}"
        
        return output_string.rstrip()
    
    def index_documents(self, directory_path: str = ".") -> None:
        """
        Index all documents in the directory by creating embeddings.
        
        Args:
            directory_path (str): Path to directory to index
        """
        print("📚 Indexing documents...")
        base_path = Path(directory_path)
        
        # Get all text files in directory and subdirectories
        text_files = []
        for ext in ['.txt', '.json', '.csv', '.pdf']:
            text_files.extend(base_path.rglob(f"*{ext}"))
        
        # Also include files without extensions that might be text
        for item in base_path.rglob("*"):
            if item.is_file() and item.suffix == "" and item.stat().st_size < 1000000:  # < 1MB
                text_files.append(item)
        
        for file_path in text_files:
            try:
                # Skip binary files and very large files
                if file_path.stat().st_size > 5000000:  # Skip files > 5MB
                    continue
                    
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Only index files with reasonable content
                if len(content.strip()) > 10:  # At least 10 characters of content
                    relative_path = str(file_path.relative_to(base_path))
                    self.file_contents[relative_path] = content
                    
                    # Create embedding for the file content
                    print(f"   📄 Indexing: {relative_path}")
                    embedding = self.get_embedding(content[:4000])  # Limit content for embedding
                    self.document_embeddings[relative_path] = {
                        'embedding': embedding,
                        'content': content
                    }
                    
            except Exception as e:
                print(f"   ✗ Error indexing {file_path}: {e}")
        
        print(f"✅ Indexed {len(self.document_embeddings)} documents")
    
    def find_relevant_documents(self, question: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Find the most relevant documents for a question using semantic search.
        
        Args:
            question (str): User question
            top_k (int): Number of top documents to return
            
        Returns:
            List[Tuple[str, float, str]]: List of (filename, similarity_score, content) tuples
        """
        if not self.document_embeddings:
            return []
        
        print("🔍 Finding relevant documents...")
        question_embedding = self.get_embedding(question)
        
        # Calculate similarity scores
        similarities = []
        for filename, doc_data in self.document_embeddings.items():
            similarity = self.cosine_similarity(question_embedding, doc_data['embedding'])
            similarities.append((filename, similarity, doc_data['content']))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def ask_ollama_question(self, question: str, context: str) -> str:
        """
        Ask Ollama a question with context.
        
        Args:
            question (str): User question
            context (str): Context from relevant documents
            
        Returns:
            str: Answer from Ollama
        """
        try:
            prompt = f"""Based on the following documents, please answer the user's question.

DOCUMENTS:
{context}

USER QUESTION: {question}

Please provide a comprehensive answer based on the document contents. 
If the answer cannot be found in the documents, please state that clearly.
Be specific and cite which documents contain the relevant information."""

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": "gemma2:2b",  # You can change this to any model you have
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["response"]
            
        except Exception as e:
            return f"Error asking Ollama: {str(e)}"
    
    def ask_about_documents(self, user_question: str, directory_path: str = ".") -> str:
        """
        Main function to answer questions about documents using Ollama embeddings.
        
        Args:
            user_question (str): User's question
            directory_path (str): Base directory path
            
        Returns:
            str: Answer to the user's question
        """
        try:
            # Index documents if not already indexed
            if not self.document_embeddings:
                self.index_documents(directory_path)
            
            if not self.document_embeddings:
                return "No documents were found to index in the directory."
            
            # Find relevant documents using semantic search
            relevant_docs = self.find_relevant_documents(user_question)
            
            if not relevant_docs:
                return "No relevant documents found for your question."
            
            print(f"📑 Found {len(relevant_docs)} relevant documents:")
            for i, (filename, score, content) in enumerate(relevant_docs):
                print(f"   {i+1}. {filename} (similarity: {score:.3f})")
            
            # Prepare context from relevant documents
            context_parts = []
            for filename, score, content in relevant_docs:
                # Use truncated content to avoid token limits
                truncated_content = content[:3000] + "..." if len(content) > 3000 else content
                context_parts.append(f"--- {filename} (relevance: {score:.3f}) ---\n{truncated_content}")
            
            context = "\n\n".join(context_parts)
            
            # Ask Ollama with the context
            print("🤔 Generating answer...")
            answer = self.ask_ollama_question(user_question, context)
            
            # Add source information
            source_files = [filename for filename, _, _ in relevant_docs]
            answer += f"\n\n📚 Sources: {', '.join(source_files)}"
            
            return answer
            
        except Exception as e:
            return f"Error processing your request: {str(e)}"

qa_system = OllamaDocumentQA(ollama_base_url="http://localhost:11434")



def setup_ollama_qa(directory_path: str = "./data/"):    
    qa_system.set_directory(directory_path)

    qa_system.index_documents(directory_path)
    print("\n🔧 Indexing documents (this may take a while)...")

def ask_ollama_qa(user_question: str = "", directory_path = "./data/"):
    answer = qa_system.ask_about_documents(user_question, directory_path)
    return answer
    

# # Example usage
# if __name__ == "__main__":
#     # Example 1: One-time question
#     def example_single_question(question):
#         qa_system = OllamaDocumentQA()
#         directory = "data"
#         listing = qa_system.get_directory_listing_string(directory)
        
#         answer = qa_system.ask_about_documents(listing, question, directory)
#         print("Question:", question)
#         print("Answer:", answer)
    
#     # Example 2: Interactive mode
#     interactive_ollama_qa('data')  # Use current directory
    
#     # Example 3: Custom Ollama URL
#     # interactive_ollama_qa("/path/to/documents", "http://192.168.1.100:11434")