# ============================================================
#  Sentiment Analysis App — Full Emotion Breakdown
#  Author  : Aarica Raj
#  GitHub  : github.com/Aaricacoding
#  Project : NLP with Transformers (7 Emotions + Positive/Negative)
# ============================================================

import gradio as gr
from transformers import pipeline

# ── 1. Load Both Models ─────────────────────────────────────
print("Loading models...")

emotion_pipeline   = pipeline(
    task  = "text-classification",
    model = "j-hartmann/emotion-english-distilroberta-base",
    top_k = None
)

sentiment_pipeline = pipeline(
    task  = "sentiment-analysis",
    model = "distilbert-base-uncased-finetuned-sst-2-english",
    top_k = None
)

print("All models loaded successfully!")

EMOJI_MAP = {
    "joy"      : "😊 Joy",
    "sadness"  : "😢 Sadness",
    "anger"    : "😠 Anger",
    "fear"     : "😨 Fear",
    "surprise" : "😲 Surprise",
    "disgust"  : "🤢 Disgust",
    "neutral"  : "😐 Neutral",
    "POSITIVE" : "👍 Positive",
    "NEGATIVE" : "👎 Negative",
}


# ── 2. Analysis Function ────────────────────────────────────
def analyze(text):
    if not text or text.strip() == "":
        return "Please enter some text!", "", {}, {}

    # Emotion scores
    emotion_results  = emotion_pipeline(text[:512])[0]
    emotion_scores   = {
        EMOJI_MAP.get(e['label'], e['label']): round(e['score'] * 100, 2)
        for e in emotion_results
    }

    # Positive/Negative scores
    sentiment_results = sentiment_pipeline(text[:512])[0]
    sentiment_scores  = {
        EMOJI_MAP.get(s['label'], s['label']): round(s['score'] * 100, 2)
        for s in sentiment_results
    }

    # Top emotion
    top_emotion    = max(emotion_results, key=lambda x: x['score'])
    top_sentiment  = max(sentiment_results, key=lambda x: x['score'])
    emotion_label  = EMOJI_MAP.get(top_emotion['label'], top_emotion['label'])
    sentiment_label= EMOJI_MAP.get(top_sentiment['label'], top_sentiment['label'])
    emotion_conf   = round(top_emotion['score'] * 100, 2)
    sentiment_conf = round(top_sentiment['score'] * 100, 2)

    summary = f"""
### 🎯 Result Summary
- **Primary Emotion:** {emotion_label} ({emotion_conf}%)
- **Overall Sentiment:** {sentiment_label} ({sentiment_conf}%)
"""

    return emotion_label, summary, emotion_scores, sentiment_scores


# ── 3. Gradio UI ────────────────────────────────────────────
with gr.Blocks(
    title="Full Emotion + Sentiment Analysis",
    theme=gr.themes.Soft(),
    css="""
        .header { text-align: center; padding: 20px 0 10px; }
        .footer { text-align: center; font-size: 12px; color: #888; margin-top: 10px; }
        .result { font-size: 22px; font-weight: bold; text-align: center; }
    """
) as demo:

    gr.HTML("""
        <div class='header'>
            <h1>🧠 Full Emotion & Sentiment Analyzer</h1>
            <p>Get complete breakdown — 7 Emotions + Positive/Negative for any text!</p>
            <p><i>Models: DistilRoBERTa (Emotions) + DistilBERT (Sentiment)</i></p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            text_input  = gr.Textbox(
                label       = "Enter Text",
                placeholder = "Type any sentence, review, comment...",
                lines       = 5
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

            gr.Examples(
                examples = [
                    ["I absolutely love this! Best thing ever!"],
                    ["This is terrible. I am so angry and disappointed."],
                    ["The movie was okay, nothing special but not terrible either."],
                    ["I am so scared about tomorrow's exam."],
                    ["Oh wow! I never expected that to happen!"],
                    ["This food smells disgusting, I cannot eat it."],
                    ["I went to the store and bought groceries today."],
                ],
                inputs = [text_input],
                label  = "Try these examples"
            )

        with gr.Column(scale=1):
            top_result     = gr.Textbox(
                label       = "Primary Emotion Detected",
                elem_classes= ["result"],
                interactive = False
            )
            summary_output = gr.Markdown()
            gr.Markdown("### 😊 Emotion Breakdown (7 Emotions)")
            emotion_scores = gr.Label(
                label           = "Emotion Scores",
                num_top_classes = 7
            )
            gr.Markdown("### 👍 Sentiment Breakdown (Positive / Negative)")
            sentiment_scores = gr.Label(
                label           = "Sentiment Scores",
                num_top_classes = 2
            )

    gr.HTML("""
        <div class='footer'>
            Built by <b>Aarica Raj</b> &nbsp;|&nbsp;
            <a href='https://github.com/Aaricacoding' target='_blank'>GitHub</a> &nbsp;|&nbsp;
            Powered by HuggingFace Transformers + Gradio
        </div>
    """)

    analyze_btn.click(
        fn      = analyze,
        inputs  = [text_input],
        outputs = [top_result, summary_output, emotion_scores, sentiment_scores]
    )

    text_input.submit(
        fn      = analyze,
        inputs  = [text_input],
        outputs = [top_result, summary_output, emotion_scores, sentiment_scores]
    )


# ── 4. Launch ───────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share       = False,
        server_name = "0.0.0.0",
        server_port = 7861,
        show_error  = True
    )
