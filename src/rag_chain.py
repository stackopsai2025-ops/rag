from typing import List, Tuple, Dict
from groq import Groq


def answer_with_rag(
    client: Groq,
    query: str,
    contexts: List[Tuple[str, Dict]],
    model: str,
    strict: bool = True,
) -> str:
    """
    Assemble prompt with retrieved context and call Groq.
    Prints sources so viewer sees grounding.
    """
    context_text = "\n\n".join(
        [f"[source: {m.get('path','?')}]\n{t}" for t, m in contexts]
    )

    system = (
        "You are a senior engineer assistant.\n"
        "Use ONLY the provided context.\n"
        "If the answer is not explicitly in the context, say:\n"
        "'I don't know based on the provided context.'\n"
        "When you answer, include a 'Sources:' line listing file paths used."
        if strict else
        "You are a helpful assistant."
    )

    user = f"""Context:
{context_text}

Question: {query}

Return format:
- Answer: <your answer>
- Sources: <comma-separated file paths, or 'None'>
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return resp.choices[0].message.content
