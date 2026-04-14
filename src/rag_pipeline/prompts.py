"""Centralized prompt templates used by the RAG pipeline."""

# Strict shared prompt for all models (fair comparison & anti-hallucination).
STRICT_RAG_PROMPT = (
    "You MUST answer ONLY based on the provided context.\n\n"
    "If the answer is not in the context, say \"I don't know\".\n\n"
    "Do NOT use prior knowledge.\n"
    "Do NOT hallucinate.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n"
)


# Controlled-reasoning prompt:
# - Allows simple, bounded reasoning ONLY when logically derivable from context.
# - Keeps strict abstention and anti-hallucination guardrails.
CONTROLLED_REASONING_RAG_PROMPT = (
    "You must answer based ONLY on the provided context.\n\n"
    "You are REQUIRED to perform reasoning when needed.\n\n"
    "---\n\n"
    "INSTRUCTIONS:\n\n"
    "1. Identify any date or timestamp in the context.\n\n"
    "2. Identify time expressions such as \"tomorrow\", \"today\", etc.\n\n"
    "3. Apply the following rules:\n\n"
    "   * \"tomorrow\" = next day of the given date\n"
    "   * Use the timestamp in the context as the reference date\n\n"
    "4. Combine the inferred date with the mentioned time (if available).\n\n"
    "5. Output the final answer.\n\n"
    "---\n\n"
    "RULES:\n\n"
    "* You MUST use only the context.\n"
    "* You MUST perform reasoning if time expressions exist.\n"
    "* You MUST NOT ignore \"tomorrow\" or similar expressions.\n"
    "* If partial information exists, provide the best possible answer.\n\n"
    "---\n\n"
    "OUTPUT FORMAT (STRICT):\n\n"
    "Answer: <final answer>\n"
    "Explanation: <one sentence explaining the reasoning>\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Additional Rules for Time:\n\n"
    "* \"tomorrow\" = next day of the given date\n"
    "* Use the timestamp as reference\n"
)
