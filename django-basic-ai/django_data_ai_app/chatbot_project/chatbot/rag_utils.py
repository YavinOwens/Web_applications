import os
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ensure NLTK data is downloaded
try:
    nltk.download('all', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

class DocumentProcessor:
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        
    def process_document(self, file_path):
        """Process a document and add it to the collection"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into sentences
            sentences = sent_tokenize(text)
            
            # Add each sentence as a chunk with metadata
            for i, sentence in enumerate(sentences):
                self.documents.append({
                    'chunk': sentence,
                    'file_path': file_path,
                    'position': f"sentence {i+1}"
                })
            
            # Update document vectors
            if self.documents:
                chunks = [doc['chunk'] for doc in self.documents]
                self.document_vectors = self.vectorizer.fit_transform(chunks)
        
        except Exception as e:
            print(f"Error processing document: {str(e)}")
    
    def get_relevant_chunks(self, query, top_k=3):
        """Get the most relevant chunks for a query"""
        if not self.documents or self.document_vectors is None:
            return []
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Get top k chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                chunk = self.documents[idx].copy()
                chunk['similarity'] = similarities[idx]
                relevant_chunks.append(chunk)
            
            return relevant_chunks
        
        except Exception as e:
            print(f"Error getting relevant chunks: {str(e)}")
            return []

def prepare_messages_with_context(user_input, relevant_chunks, system_prompt):
    """Prepare messages for OpenAI API with context from relevant chunks"""
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add context from relevant chunks
    if relevant_chunks:
        context = "Here are some relevant passages:\n\n"
        for chunk in relevant_chunks:
            context += f"From {chunk['file_path']} ({chunk['position']}):\n{chunk['chunk']}\n\n"
        
        messages.append({
            "role": "system",
            "content": f"{context}\nPlease use this information to help answer the user's question."
        })
    
    # Add user's question
    messages.append({"role": "user", "content": user_input})
    
    return messages 