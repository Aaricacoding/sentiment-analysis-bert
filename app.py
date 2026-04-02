# ============================================================
#  Sentiment Analysis App — BERT + Gradio
#  Author  : Aarica Raj
#  GitHub  : github.com/Aaricacoding
#  Project : NLP with Transformers (BERT Fine-tuning)
# ============================================================

import gradio as gr
from transformers import pipeline
import time

# ── 1. Load Pre-trained Sentiment Model ────────────────────
print("Loading BERT Sentiment Analysis model...")

sentiment_pipeline = pipeline(
    task   = "sentiment-analysis",
    model  = "distilbert-base-uncased-finetuned-sst-2-english",
    top_k  = None  # returns scores for ALL labels
)

print("Model loaded successfully!")


# ── 2. Sentiment Analysis Function ─────────────────────────
def analyze_sentiment(text):
    """
    Analyze the sentiment of input text using BERT.

    Args:
        text: Input text from user

    Returns:
        tuple: (label, confidence, all_scores, emoji)
    """
    if not text or text.strip() == "":
        return "⚠️ Please enter some text!", "", {}, ""

    # Run through BERT pipeline
    results = sentiment_pipeline(text[:512])[0]  # limit to 512 tokens

    # Extract scores
    scores = {item['label']: round(item['score'] * 100, 2) for item in results}

    # Get top result
    top     = max(results, key=lambda x: x['score'])
    label   = top['label']
    confidence = round(top['score'] * 100, 2)

    # Emoji based on sentiment
    if label == "POSITIVE":
        emoji   = "😊 POSITIVE"
        message = f"The text has a **positive** sentiment with **{confidence}%** confidence!"
    else:
        emoji   = "😞 NEGATIVE"
        message = f"The text has a **negative** sentiment with **{confidence}%** confidence!"

    return emoji, message, scores


# ── 3. Gradio UI ────────────────────────────────────────────
with gr.Blocks(
    title="Sentiment Analysis - BERT",
    theme=gr.themes.Soft(),
    css="""
        .header { text-align: center; padding: 20px 0 10px; }
        .footer { text-align: center; font-size: 12px; color: #888; margin-top: 10px; }
        .result { font-size: 20px; font-weight: bold; text-align: center; }
    """
) as demo:

    # Header
    gr.HTML("""
        <div class='header'>
            <h1>🧠 Sentiment Analysis using BERT</h1>
            <p>Type any text and let the AI detect if it's Positive or Negative!</p>
            <p><i>Model: DistilBERT fine-tuned on SST-2 &nbsp;|&nbsp;
               Architecture: BERT Transformer</i></p>
        </div>
    """)

    with gr.Row():
        # Left — Input
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label       = "Enter Text",
                placeholder = "Type a sentence, review, or any text here...",
                lines       = 5
            )
            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")

            gr.Examples(
                examples = [
                    ["I absolutely love this product! It works perfectly and exceeded my expectations."],
                    ["This is the worst experience I have ever had. Completely disappointed."],
                    ["The movie was okay, nothing special but not terrible either."],
                    ["I am so happy today! Everything is going great in my life."],
                    ["I hate waiting in long queues. It wastes so much of my time."],
                ],
                inputs = [text_input],
                label  = "Try these examples"
            )

        # Right — Output
        with gr.Column(scale=1):
            sentiment_label = gr.Textbox(
                label       = "Sentiment Result",
                elem_classes= ["result"],
                interactive = False
            )
            message_output = gr.Markdown(label="Analysis")
            score_output   = gr.Label(
                label      = "Confidence Scores (%)",
                num_top_classes = 2
            )

    # Footer
    gr.HTML("""
        <div class='footer'>
            Built by <b>Aarica Raj</b> &nbsp;|&nbsp;
            <a href='https://github.com/Aaricacoding' target='_blank'>GitHub</a> &nbsp;|&nbsp;
            Powered by HuggingFace Transformers + Gradio
        </div>
    """)

    # Wire button
    analyze_btn.click(
        fn      = analyze_sentiment,
        inputs  = [text_input],
        outputs = [sentiment_label, message_output, score_output]
    )

    # Also trigger on Enter key
    text_input.submit(
        fn      = analyze_sentiment,
        inputs  = [text_input],
        outputs = [sentiment_label, message_output, score_output]
    )


# ── 4. Launch ───────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share       = True,
        server_name = "0.0.0.0",
        server_port = 7861,
        show_error  = True
    )
