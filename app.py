# ============================================================
#  Sentiment Analysis App — Full Emotion Breakdown
#  Author  : Aarica Raj
#  GitHub  : github.com/Aaricacoding
# ============================================================

import gradio as gr
from transformers import pipeline

print("Loading models...")

emotion_pipeline = pipeline(
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

def analyze(text):
    if not text or text.strip() == "":
        return "Please enter some text!", "", "", ""

    emotion_results   = emotion_pipeline(text[:512])[0]
    sentiment_results = sentiment_pipeline(text[:512])[0]

    # Top results
    top_emotion    = max(emotion_results, key=lambda x: x['score'])
    top_sentiment  = max(sentiment_results, key=lambda x: x['score'])
    emotion_label  = EMOJI_MAP.get(top_emotion['label'], top_emotion['label'])
    sentiment_label= EMOJI_MAP.get(top_sentiment['label'], top_sentiment['label'])
    emotion_conf   = round(top_emotion['score'] * 100, 2)
    sentiment_conf = round(top_sentiment['score'] * 100, 2)

    # Format emotion scores as text
    emotion_text = "### 😊 Emotion Breakdown\n"
    for e in sorted(emotion_results, key=lambda x: x['score'], reverse=True):
        label = EMOJI_MAP.get(e['label'], e['label'])
        score = round(e['score'] * 100, 2)
        bar   = "█" * int(score // 5)
        emotion_text += f"**{label}**: {score}% {bar}\n\n"

    # Format sentiment scores as text
    sentiment_text = "### 👍 Sentiment Breakdown\n"
    for s in sorted(sentiment_results, key=lambda x: x['score'], reverse=True):
        label = EMOJI_MAP.get(s['label'], s['label'])
        score = round(s['score'] * 100, 2)
        bar   = "█" * int(score // 5)
        sentiment_text += f"**{label}**: {score}% {bar}\n\n"

    summary = f"**Primary Emotion:** {emotion_label} ({emotion_conf}%)  |  **Overall Sentiment:** {sentiment_label} ({sentiment_conf}%)"

    return emotion_label, summary, emotion_text, sentiment_text


with gr.Blocks(title="Emotion & Sentiment Analyzer") as demo:

    gr.Markdown("# 🧠 Full Emotion & Sentiment Analyzer")
    gr.Markdown("Type any text and get complete breakdown — 7 Emotions + Positive/Negative!")
    gr.Markdown("*Models: DistilRoBERTa (Emotions) + DistilBERT (Sentiment)*")

    with gr.Row():
        with gr.Column():
            text_input  = gr.Textbox(
                label       = "Enter Text",
                placeholder = "Type any sentence, review, comment...",
                lines       = 5
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")
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

        with gr.Column():
            top_result      = gr.Textbox(label="Primary Emotion Detected", interactive=False)
            summary_output  = gr.Markdown()
            emotion_scores  = gr.Markdown()
            sentiment_scores= gr.Markdown()

    gr.Markdown("Built by **Aarica Raj** | [GitHub](https://github.com/Aaricacoding)")

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

if __name__ == "__main__":
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860
    )
