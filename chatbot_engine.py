"""
RAG Chatbot Engine for Student FAQs
=====================================

Architecture:
  RETRIEVE   → TF-IDF + cosine similarity finds the top-K most relevant
               FAQ entries from the local CSV dataset (fast, free, local).
  AUGMENT    → Retrieved FAQs are injected into the system prompt so the
               LLM answers only from that context (no hallucination).
  GENERATE   → Groq API (Llama 3 model) produces a natural, conversational
               answer. Groq is used because it has a generous free tier
               (14,400 requests/day), requires no credit card, and its
               SDK is simple and reliable.

Why Groq for an undergraduate project?
  - Free tier: 14,400 req/day, no billing info needed
  - Fast: responses in ~0.3s (Groq runs on custom LPU hardware)
  - Simple API: OpenAI-compatible, easy to understand and explain
  - Academic credibility: uses Meta's open-source Llama 3 model
  - Get a key at: https://console.groq.com
"""

import re
import pandas as pd
from groq import Groq, AuthenticationError, RateLimitError, APIConnectionError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOP_K = 4
MODEL = "openai/gpt-oss-20b"   # fast, free, great for Q&A tasks
REQUIRED_COLUMNS = {"question", "answer"}

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful and friendly university student support assistant.

Answer the student's question using ONLY the FAQ entries provided below.
Do not use any knowledge outside what is given in the FAQ context.
If the context does not cover the question, say so honestly and suggest
the student contact the relevant university office directly.

Keep your answer clear, concise, and friendly. You may combine information
from multiple FAQ entries if needed, but never invent details.

--- FAQ CONTEXT ---
{context}
--- END OF CONTEXT ---\
"""


class RAGChatbot:
    def __init__(self):
        self.df               = None
        self.vectorizer       = None
        self.question_vectors = None
        self.is_ready         = False
        self._client          = None   # Groq client

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def configure(self, api_key: str):
        """
        Initialise and validate the Groq client.
        Makes a tiny test call to catch bad keys immediately.
        """
        client = Groq(api_key=api_key.strip())
        # Quick validation — send one token, just to confirm the key works.
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        self._client = client

    def load_dataframe(self, df: pd.DataFrame):
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {', '.join(missing)}. "
                "The file needs at least 'question' and 'answer' columns."
            )

        df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
        df["question"] = df["question"].astype(str).str.strip()
        df["answer"]   = df["answer"].astype(str).str.strip()
        df = df[(df["question"] != "") & (df["answer"] != "")].reset_index(drop=True)

        if "category" not in df.columns:
            df["category"] = "General"
        else:
            df["category"] = df["category"].fillna("General").astype(str).str.strip()

        if len(df) == 0:
            raise ValueError("No usable rows found after cleaning the dataset.")

        self.df = df
        cleaned = df["question"].apply(self._clean)
        self.vectorizer       = TfidfVectorizer(stop_words="english")
        self.question_vectors = self.vectorizer.fit_transform(cleaned)
        self.is_ready = True

    def load_csv(self, filepath: str):
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Could not read CSV: {e}")
        self.load_dataframe(df)

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text).lower().strip())

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """Return the top-K most relevant FAQ entries for the query."""
        if not self.is_ready or not query.strip():
            return []
        vec  = self.vectorizer.transform([self._clean(query)])
        sims = cosine_similarity(vec, self.question_vectors).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if sims[idx] > 0:
                row = self.df.iloc[idx]
                results.append({
                    "question":   row["question"],
                    "answer":     row["answer"],
                    "category":   row["category"],
                    "similarity": float(sims[idx]),
                })
        return results

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #

    def _build_context(self, retrieved: list) -> str:
        if not retrieved:
            return "(No relevant FAQ entries found for this query.)"
        lines = []
        for i, item in enumerate(retrieved, 1):
            lines.append(
                f"[{i}] Category: {item['category']}\n"
                f"    Q: {item['question']}\n"
                f"    A: {item['answer']}"
            )
        return "\n\n".join(lines)

    def _build_messages(self, user_message: str, context_str: str,
                        chat_history: list) -> list:
        """
        Build the messages list for the Groq chat completion call.

        Structure:
          [system]          ← FAQ context injected here
          [user] [assistant] ... ← previous turns
          [user]            ← current question
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context_str)}
        ]
        for turn in chat_history:
            role = "assistant" if turn["role"] == "assistant" else "user"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate_response(self, user_message: str, chat_history: list) -> dict:
        """
        Full RAG pipeline for one conversational turn.

        Returns dict: answer (str), retrieved (list), context_str (str)
        """
        if not self._client:
            return {
                "answer": (
                    "⚠️ API key not set. Go to the ⚙️ Settings tab, "
                    "paste your Groq API key, and click **Save**."
                ),
                "retrieved": [], "context_str": "",
            }

        # STEP 1 — RETRIEVE
        retrieved   = self.retrieve(user_message)
        context_str = self._build_context(retrieved)

        # STEP 2 — AUGMENT + STEP 3 — GENERATE
        messages = self._build_messages(user_message, context_str, chat_history)
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            answer = response.choices[0].message.content
        except AuthenticationError:
            answer = "⚠️ Invalid Groq API key. Please check it in the ⚙️ Settings tab."
        except RateLimitError:
            answer = "⚠️ Groq rate limit reached. Wait a moment and try again."
        except APIConnectionError as e:
            answer = f"⚠️ Could not connect to Groq: {e}"
        except Exception as e:
            answer = f"⚠️ Unexpected error: {e}"

        return {"answer": answer, "retrieved": retrieved, "context_str": context_str}

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def list_categories(self):
        return sorted(self.df["category"].unique().tolist()) if self.is_ready else []

    def num_faqs(self):
        return len(self.df) if self.is_ready else 0