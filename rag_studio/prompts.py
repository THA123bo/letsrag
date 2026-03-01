"""System prompts for the RAG assistant."""


SYSTEM_PROMPT = (
    "You are an expert internal technical assistant. "
    "Your objective is to provide highly precise answers using ONLY the provided context fragments.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Read all fragments carefully. Information might be split across them.\n"
    "2. THINK STEP BY STEP: Identify the core facts before answering.\n"
    "3. SYNTHESIZE the logic: pay attention to cause-and-effect, versions, dates, and constraints.\n"
    "4. DO NOT invent or hallucinate any data, numbers, or rules not explicitly stated.\n"
    "5. Be CONCISE: answer in 3-5 sentences maximum. Do NOT add headers, bullet introductions, or preamble like 'Based on the fragments...'\n"
    "6. If the context fragments do not contain the answer, reply ONLY with: 'I do not have enough information'."
)


def build_user_prompt(context_chunks: list[str], question: str) -> str:
    """Builds the user prompt combining document context and the user's question.

    Args:
        context_chunks (list[str]): A list of retrieved text fragments.
        question (str): The user's question.

    Returns:
        str: The formatted prompt containing numbered fragments and the question.
    """
    numbered_chunks = [f"[Fragment {i}]\n{chunk}" for i, chunk in enumerate(context_chunks, 1)]
    context = "\n\n---\n\n".join(numbered_chunks)
    
    return (
        f"Below you will find {len(context_chunks)} internal document fragments.\n"
        f"Analyze ALL fragments and answer the question by combining the relevant information.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )
