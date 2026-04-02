# ============================================================
#  Sentiment Analysis App — Multi-Emotion Detection
#  Author  : Aarica Raj
#  GitHub  : github.com/Aaricacoding
#  Project : NLP with Transformers (7 Emotions!)
# ============================================================

import gradio as gr
from transformers import pipeline

# ── 1. Load Multi-Emotion Model ────────────────────────────
print("Loading Emotion Detection model...")

sentiment_pipeline = pipeline(
    task  = "text-classification",
    model = "j-hartmann/emotion-english-distilroberta-base",
    top_k = None
)

print("Model loaded successfully!")

# Emoji map for each emotion
EMOJI_MAP = {
    "joy"      : "😊 JOY",
    "sadness"  : "😢 SADNESS",
    "anger"    : "😠 ANGER",
    "fear"     : "😨 FEAR",
    "surprise" : "😲 SURPRISE",
    "disgust"  : "🤢 DISGUST",
    "neutral"  : "😐 NEUTRAL",
}


# ── 2. Emotion Detection Function ──────────────────────────
def analyze_sentiment(text):
    if not text or text.strip() == "":
        return "Please enter some text!", "", {}

    results    = sentiment_pipeline(text[:512])[0]
    scores     = {item['label'].capitalize(): round(item['score'] * 100, 2) for item in results}
    top        = max(results, key=lambda x: x['score'])
    label      = top['label'].lower()
    confidence = round(top['score'] * 100, 2)
    emoji      = EMOJI_MAP.get(label, f"🔍 {label.upper()}")
    message    = f"The text expresses **{label.upper()}** with **{confidence}%** confidence!"

    return emoji, message, scores


# ── 3. Gradio UI ────────────────────────────────────────────
with gr.Blocks(
    title="Emotion Detection - Multi Sentiment",
    theme=gr.themes.Soft(),
    css="""
        .header { text-align: center; padding: 20px 0 10px; }
        .footer { text-align: center; font-size: 12px; color: #888; margin-top: 10px; }
        .result { font-size: 20px; font-weight: bold; text-align: center; }
    """
) as demo:

    gr.HTML("""
        <div class='header'>
            <h1>🧠 Emotion Detection using BERT</h1>
            <p>Type any text and let AI detect the emotion — Joy, Sadness, Anger, Fear, Surprise, Disgust or Neutral!</p>
            <p><i>Model: DistilRoBERTa &nbsp;|&nbsp; 7 Emotions &nbsp;|&nbsp; HuggingFace Transformers</i></p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label       = "Enter Text",
                placeholder = "Type a sentence, review, or any text here...",
                lines       = 5
            )
            analyze_btn = gr.Button("🔍 Detect Emotion", variant="primary", size="lg")

            gr.Examples(
                examples = [
                    ["I absolutely love this! It made my day so much better!"],
                    ["This is the worst experience I have ever had. I am so angry!"],
                    ["The movie was okay, nothing special but not terrible either."],
                    ["I am so scared, I don't know what will happen next."],
                    ["Oh wow! I never expected that to happen, what a surprise!"],
                    ["This food smells disgusting, I cannot eat it."],
                    ["I went to the store and bought some groceries today."],
                ],
                inputs = [text_input],
                label  = "Try these examples"
            )

        with gr.Column(scale=1):
            sentiment_label = gr.Textbox(
                label       = "Detected Emotion",
                elem_classes= ["result"],
                interactive = False
            )
            message_output = gr.Markdown(label="Analysis")
            score_output   = gr.Label(
                label           = "All Emotion Scores (%)",
                num_top_classes = 7
            )

    gr.HTML("""
        <div class='footer'>
            Built by <b>Aarica Raj</b> &nbsp;|&nbsp;
            <a href='https://github.com/Aaricacoding' target='_blank'>GitHub</a> &nbsp;|&nbsp;
            Powered by HuggingFace Transformers + Gradio
        </div>
    """)

    analyze_btn.click(
        fn      = analyze_sentiment,
        inputs  = [text_input],
        outputs = [sentiment_label, message_output, score_output]
    )

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
