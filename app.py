"""
RAG Chatbot for Student FAQs — Gradio App
==========================================
Undergraduate Final Year Project

Run with:  python app.py
Then open: http://127.0.0.1:7860
"""

import os
import gradio as gr
from chatbot_engine import RAGChatbot

DEFAULT_FAQ_PATH = os.path.join(os.path.dirname(__file__), "faqs.csv")

bot = RAGChatbot()
bot.load_csv(DEFAULT_FAQ_PATH)


# ------------------------------------------------------------------ #
#  Callbacks                                                           #
# ------------------------------------------------------------------ #

def chat_respond(user_message, history):
    if not user_message or not user_message.strip():
        return ""
    result = bot.generate_response(user_message, history or [])
    if result["retrieved"]:
        sources = "\n".join(
            f"  • [{r['category']}] {r['question']} ({r['similarity']:.0%} match)"
            for r in result["retrieved"]
        )
        footer = f"\n\n---\n📚 **Retrieved FAQ sources:**\n{sources}"
    else:
        footer = ""
    return result["answer"] + footer


def save_api_key(api_key: str):
    if not api_key or not api_key.strip():
        return "<div class='status-box status-warn'>⚠️ Please enter your Groq API key.</div>"
    try:
        bot.configure(api_key)
        return "<div class='status-box status-ok'>✅ Groq API key saved. Switch to the <b>Chat</b> tab to start.</div>"
    except Exception as e:
        return f"<div class='status-box status-warn'>⚠️ {e}</div>"


def upload_new_faqs(file):
    if file is None:
        return "<div class='status-box status-warn'>⚠️ No file uploaded.</div>", _dataset_info()
    try:
        bot.load_csv(file)
        return f"<div class='status-box status-ok'>✅ Loaded <b>{bot.num_faqs()}</b> FAQs successfully.</div>", _dataset_info()
    except ValueError as e:
        return f"<div class='status-box status-warn'>⚠️ {e}</div>", _dataset_info()


def reset_to_default():
    try:
        bot.load_csv(DEFAULT_FAQ_PATH)
        return "<div class='status-box status-ok'>✅ Reset to default FAQ dataset.</div>", _dataset_info()
    except Exception as e:
        return f"<div class='status-box status-warn'>⚠️ {e}</div>", _dataset_info()


def _dataset_info():
    cats = ", ".join(bot.list_categories())
    return (
        f"<div class='dataset-card'>"
        f"<span class='dataset-count'>{bot.num_faqs()}</span> questions loaded"
        f"<div class='dataset-cats'>📁 {cats}</div>"
        f"</div>"
    )


# ------------------------------------------------------------------ #
#  Theme & Styling                                                     #
# ------------------------------------------------------------------ #

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_label_text_weight="500",
)

css_file_path = os.path.join(os.path.dirname(__file__), "styles.css")
with open(css_file_path, "r", encoding="utf-8") as f:
    CUSTOM_CSS = f.read()



# ------------------------------------------------------------------ #
#  UI                                                                  #
# ------------------------------------------------------------------ #

with gr.Blocks(title="ASKA — STU Student Assistant") as demo:

    gr.HTML(
        """
        <div class="hero">
            <h1>🎓 ASKA — The STU Student Assistant</h1>
            <h3 class="tagline">Your Intelligent Guide to Sunyani Technical University</h3>
            <div class="badges">
                <span class="badge">🎓 Programmes</span>
                <span class="badge">📋 Admissions</span>
                <span class="badge">💰 Fees</span>
                <span class="badge">📞 Contacts</span>
            </div>
            <div class="setup-note">
                ⚙️ <b>First time here?</b> Open the <b>Settings</b> tab, paste your Groq API key,
                and click <b>Save key</b>. Then return here to start chatting.
            </div>
        </div>
        """
    )

    # ── Tab 1: Chat ──────────────────────────────────────────────────
    with gr.Tab("💬 Chat"):
        gr.Markdown(
            "Ask ASKA anything about **admissions, programmes, fees, or contacts** at STU. "
            "Answers are grounded in the university's official FAQ knowledge base."
        )
        gr.ChatInterface(
            fn=chat_respond,
            chatbot=gr.Chatbot(
                height=460,
                avatar_images=(None, "🎓"),
                placeholder="**Ask me anything about STU admissions, programmes, or fees.**",
            ),
            textbox=gr.Textbox(
                placeholder="Type your question here…",
                container=False,
                scale=7,
            ),
            examples=[
                "What programmes does STU offer?",
                "How do I apply for admission to STU?",
                "How much is the application form?",
                "What are the entry requirements for HND?",
                "How do I contact the admissions office?",
                "When is the mature applicants exam?",
            ],
            title=None,
        )

    # ── Tab 2: Settings ──────────────────────────────────────────────
    with gr.Tab("⚙️ Settings"):
        with gr.Group(elem_classes="section-card"):
            gr.Markdown("#### 🔑 Groq API Key")
            gr.Markdown(
                "Get a free key (no credit card required) at "
                "[console.groq.com](https://console.groq.com). "
                "The key is kept in memory only — it is never written to disk."
            )
            with gr.Row():
                api_key_input = gr.Textbox(
                    placeholder="Paste your Groq API key here (starts with gsk_...)",
                    type="password", label="API Key", scale=4,
                )
                save_btn = gr.Button("💾 Save key", variant="primary", scale=1)
            key_status = gr.HTML()

        with gr.Group(elem_classes="section-card"):
            gr.Markdown("#### 📂 FAQ Knowledge Base")
            dataset_info = gr.HTML(_dataset_info())
            gr.Markdown(
                "Upload a CSV with **`question`**, **`answer`**, and optionally "
                "**`category`** columns to use your own institution's FAQs."
            )
            faq_file = gr.File(label="Upload FAQ CSV", file_types=[".csv"], type="filepath")
            with gr.Row():
                load_btn  = gr.Button("📤 Load uploaded CSV", variant="primary")
                reset_btn = gr.Button("↩️ Reset to default")
            upload_status = gr.HTML()

            with gr.Accordion("📄 Expected CSV format", open=False):
                gr.Markdown(
                    """
                    ```
                    question,answer,category
                    How do I pay my fees?,"Pay via the online portal.",Fees
                    When do exams start?,"Exams begin in week 12.",Examinations
                    ```
                    """
                )

        save_btn.click(save_api_key, inputs=[api_key_input], outputs=[key_status])
        load_btn.click(upload_new_faqs, inputs=[faq_file], outputs=[upload_status, dataset_info])
        reset_btn.click(reset_to_default, inputs=[], outputs=[upload_status, dataset_info])

    # ── Tab 3: How it works ──────────────────────────────────────────
    with gr.Tab("ℹ️ How It Works"):
        with gr.Group(elem_classes="section-card"):
            gr.Markdown(
                """
                ## RAG Architecture

                This chatbot uses **Retrieval-Augmented Generation (RAG)**:

                ```
                Student question
                      │
                      ▼
                ┌──────────────────────────────────────┐
                │  1. RETRIEVE                          │
                │  TF-IDF + Cosine Similarity           │
                │  → top-4 relevant FAQs from CSV       │
                └─────────────────┬────────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────────┐
                │  2. AUGMENT                           │
                │  Inject FAQs + chat history           │
                │  into the Llama 3 system prompt       │
                └─────────────────┬────────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────────┐
                │  3. GENERATE                          │
                │  Groq (Llama 3.1 8B) writes a         │
                │  natural answer from context only     │
                └─────────────────┬────────────────────┘
                                  │
                                  ▼
                         Answer + FAQ sources shown
                ```
                """
            )

        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown(
                        """
                        #### ⚡ Why Groq + Llama 3?
                        - **Free tier** — 14,400 requests/day, no credit card
                        - **Fast** — ~0.3s responses on Groq's LPU hardware
                        - **Open-source model** — Meta's Llama 3, academically credible
                        - **Reliable** — standard HTTPS, no gRPC/SDK issues
                        """
                    )
            with gr.Column():
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown(
                        """
                        #### 🎯 Why RAG over a plain LLM?
                        - **Grounded** — answers come only from your FAQ dataset
                        - **Updatable** — swap the CSV, bot instantly knows new policies
                        - **Transparent** — every reply shows retrieved FAQs
                        - **Multi-turn** — full conversation history passed each turn
                        """
                    )

if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        css=CUSTOM_CSS,
        server_name="0.0.0.0", 
        server_port=10000
    )
