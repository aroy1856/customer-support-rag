"""Prompt templates for LangGraph RAG nodes."""

# Document grading prompt
GRADE_DOCUMENT_PROMPT = """You are a grader assessing the relevance of a retrieved document to a user question.

Retrieved Document:
{document}

User Question: {question}

Instructions:
- If the document contains keywords, concepts, or information related to the user question, grade it as relevant.
- The document does not need to fully answer the question, just be topically relevant.

Provide your assessment as a JSON object with a single key "relevant" that is either "yes" or "no".

Response:"""


# Sufficiency check prompt (used implicitly - just count relevant docs)
SUFFICIENCY_MESSAGE = """I apologize, but I don't have sufficient information in my knowledge base to answer your question about: "{question}"

The documents I have don't contain relevant information for this query. Please try:
1. Rephrasing your question
2. Asking about billing, plans, roaming, activation, or Fair Usage Policy

If you need immediate assistance, please contact customer support directly."""


# Answer generation prompt
GENERATE_ANSWER_PROMPT = """You are a helpful telecom customer support assistant. Answer the user's question based ONLY on the provided context documents.

Context Documents:
{context}

User Question: {question}

Instructions:
1. Answer the question using ONLY information from the context documents above
2. Be concise but comprehensive
3. If the context doesn't contain enough information, say so
4. Do not make up or infer information not present in the documents

Answer:"""


# Answer validation prompt
VALIDATE_ANSWER_PROMPT = """You are a grader assessing whether an answer is grounded in / supported by a set of documents.

Documents:
{documents}

Answer to Validate:
{answer}

Instructions:
- Check if EVERY claim in the answer is supported by the documents
- The answer should not contain information not present in the documents
- Minor paraphrasing is acceptable as long as the meaning is preserved

Provide your assessment as a JSON object with a single key "grounded" that is either "yes" or "no".

Response:"""


# Regeneration prompt (stricter)
REGENERATE_ANSWER_PROMPT = """IMPORTANT: Your previous answer contained information not supported by the documents. 
Generate a NEW answer using ONLY the information explicitly stated in these documents.

Context Documents:
{context}

User Question: {question}

Critical Instructions:
1. Use ONLY facts explicitly stated in the documents above
2. Do NOT add any information not directly from the documents
3. If something is unclear, say "According to the documents..." rather than making assumptions
4. Keep the answer focused and factual
5. This is attempt {retry_count} of {max_retries}

Strictly Grounded Answer:"""
