"""
RAG System
Integrates encryption, retrieval, and generation for question answering
"""

from typing import List, Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG pipeline with encryption and quantization"""
    
    def __init__(self,
                 retriever,
                 llm_client,
                 prompt_template: Optional[str] = None,
                 max_context_length: int = 2000):
        """
        Initialize RAG system
        
        Args:
            retriever: Retriever instance
            llm_client: LLM client (OllamaClient or QuantizedModel)
            prompt_template: Template for QA prompt
            max_context_length: Maximum context length
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_context_length = max_context_length
        
        # Default prompt template
        if prompt_template is None:
            self.prompt_template = """Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        else:
            self.prompt_template = prompt_template
        
        logger.info("RAG system initialized")
    
    def answer_question(self, 
                       question: str,
                       top_k: int = 5,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> Dict:
        """
        Answer a question using RAG pipeline
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with answer, context, and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant chunks
        logger.info(f"Retrieving context for question: {question[:50]}...")
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        if not retrieved_chunks:
            logger.warning("No relevant chunks retrieved")
            return {
                'answer': "I couldn't find relevant information to answer this question.",
                'question': question,
                'context_chunks': [],
                'retrieval_time': retrieval_time,
                'generation_time': 0,
                'total_time': time.time() - start_time
            }
        
        # Step 2: Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)
        
        # Step 3: Build prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # Step 4: Generate answer
        logger.info("Generating answer...")
        generation_start = time.time()
        
        try:
            # Check if using Ollama or QuantizedModel
            if hasattr(self.llm_client, 'generate'):
                generation_result = self.llm_client.generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                answer = generation_result['response']
                generation_metadata = generation_result
            else:
                # Fallback
                answer = "LLM client not properly configured"
                generation_metadata = {}
            
            generation_time = time.time() - generation_start
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = f"Error generating answer: {str(e)}"
            generation_time = time.time() - generation_start
            generation_metadata = {}
        
        total_time = time.time() - start_time
        
        return {
            'answer': answer,
            'question': question,
            'context_chunks': retrieved_chunks,
            'num_chunks_retrieved': len(retrieved_chunks),
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'generation_metadata': generation_metadata
        }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from retrieved chunks
        
        Args:
            chunks: List of chunk dicts with 'text' key
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(chunks):
            text = chunk['text']
            
            # Check if adding this chunk would exceed max length
            if total_length + len(text) > self.max_context_length:
                # Try to fit partial chunk
                remaining = self.max_context_length - total_length
                if remaining > 100:  # Only add if significant space remains
                    text = text[:remaining] + "..."
                    context_parts.append(f"[{i+1}] {text}")
                break
            
            context_parts.append(f"[{i+1}] {text}")
            total_length += len(text)
        
        return "\n\n".join(context_parts)
    
    def batch_answer(self, 
                    questions: List[str],
                    top_k: int = 5,
                    temperature: float = 0.7) -> List[Dict]:
        """
        Answer multiple questions
        
        Args:
            questions: List of questions
            top_k: Number of chunks to retrieve per question
            temperature: LLM temperature
            
        Returns:
            List of answer dicts
        """
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            result = self.answer_question(question, top_k=top_k, temperature=temperature)
            results.append(result)
        
        return results
    
    def evaluate(self,
                test_questions: List[str],
                ground_truth_answers: List[str],
                top_k: int = 5) -> Dict:
        """
        Evaluate RAG system on test questions
        
        Args:
            test_questions: List of test questions
            ground_truth_answers: List of ground truth answers
            top_k: Number of chunks to retrieve
            
        Returns:
            Dict with evaluation metrics
        """
        if len(test_questions) != len(ground_truth_answers):
            raise ValueError("Number of questions must match number of answers")
        
        logger.info(f"Evaluating on {len(test_questions)} questions...")
        
        results = self.batch_answer(test_questions, top_k=top_k)
        
        # Calculate metrics
        total_retrieval_time = sum(r['retrieval_time'] for r in results)
        total_generation_time = sum(r['generation_time'] for r in results)
        total_time = sum(r['total_time'] for r in results)
        
        avg_chunks_retrieved = sum(r['num_chunks_retrieved'] for r in results) / len(results)
        
        return {
            'num_questions': len(test_questions),
            'total_time': total_time,
            'avg_time_per_question': total_time / len(test_questions),
            'total_retrieval_time': total_retrieval_time,
            'avg_retrieval_time': total_retrieval_time / len(test_questions),
            'total_generation_time': total_generation_time,
            'avg_generation_time': total_generation_time / len(test_questions),
            'avg_chunks_retrieved': avg_chunks_retrieved,
            'results': results
        }
