import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def LLM(query, context, history=None):
    """
    Generuje ostateczną odpowiedź, bezwzględnie dopasowując język do zapytania użytkownika.
    """
    MAX_CONTEXT_CHARS = 50000
    context = context[:MAX_CONTEXT_CHARS]

    if history is None:
        history = []

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert financial analyst operating a RAG system.\n\n"

                    "STRICT LANGUAGE RULE:\n"
                    "You MUST reply in the EXACT SAME language as the text provided in 'USER QUERY'.\n"
                    "- If the USER QUERY is in English -> your entire response MUST be 100% in English.\n"
                    "- If the USER QUERY is in Polish -> your entire response MUST be 100% in Polish.\n"
                    "Ignore the language of the CONTEXT. Look ONLY at the language of the USER QUERY.\n\n"

                    "CRITICAL INSTRUCTIONS:\n"
                    "- Base your answer ONLY on the provided CONTEXT (which is in English). Translate the facts accurately into the required output language.\n"
                    "- Be precise, use financial metrics and numbers from the text.\n"
                    "- Always mention which company the data belongs to using their ticker (e.g., [JPM], [AAPL]).\n"
                    "- If the context does not contain the answer, state clearly (in the query's language) that you cannot find this information."
                )
            }
        ]

        messages.extend(history)

        # Zmieniamy etykiety na angielskie (USER QUERY / CONTEXT), aby nie mylić modelu językowego
        messages.append(
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n"
                    f"---------------------------------\n"
                    f"{context}\n"
                    f"---------------------------------\n\n"
                    f"USER QUERY: {query}"
                )
            }
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,  # Jeszcze niższa temperatura dla lepszego trzymania się reguł
            messages=messages
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM error: {str(e)}"


def rewrite_query(query, history):
    """
    Tłumaczy zapytanie z dowolnego języka na angielski i zamienia je
    w zestaw profesjonalnych pojęć finansowych dla wyszukiwarki FAISS.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert cross-lingual financial search assistant. The user may ask questions "
                    "in various languages (like Polish, Spanish, or German) about SEC 10-K filings.\n\n"

                    "YOUR TASK:\n"
                    "1. Translate the core financial intent of the query into English.\n"
                    "2. Expand it into official English financial keywords, accounting terminology, and SEC Item names "
                    "that are highly likely to appear in a 10-K report.\n\n"

                    "CROSS-LINGUAL MAPPING EXAMPLES:\n"
                    "- 'ryzyka jpm' -> 'JPM, Item 1A Risk Factors, operational risks, financial challenges, liabilities, market risk'\n"
                    "- 'ile zarobili apple' -> 'AAPL, Item 7 Management's Discussion, total revenue, net income, net sales, gross profit'\n"
                    "- 'długi i płynność' -> 'liquidity and capital resources, long-term debt, credit facilities, obligations'\n\n"

                    "STRICT OUTPUT INSTRUCTION:\n"
                    "Return ONLY a flat, comma-separated list of the expanded English financial keywords. "
                    "Do NOT include any introduction, explanation, markdown formatting, or quotes. Output the raw terms directly."
                )
            }
        ]

        if history is None:
            history = []

        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": f"Translate and expand this query: {query}"
            }
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,  # Zero oznacza maksymalną precyzję
            messages=messages
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Query rewrite error: {e}")
        return query