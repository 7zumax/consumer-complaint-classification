# Gradio app: paste a complaint, get the predicted category and confidence.
# Uses the best model from models/results.json. The DistilBERT weights are too big
# for GitHub, so if they're missing it falls back to the best Keras model available.
import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import numpy as np
import torch
import gradio as gr

from text_utils import clean_text, WordTokenizer

with open("models/results.json") as f:
    info = json.load(f)
CLASSES, MAX_LEN = info["classes"], info["max_len"]

BEST = None
for name in info["results"]:  # already sorted best first
    path = "models/DistilBERT" if name == "DistilBERT" else f"models/{name}.keras"
    if os.path.exists(path):
        BEST = name
        break
print("loading", BEST)

if BEST == "DistilBERT":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_tok = AutoTokenizer.from_pretrained("models/DistilBERT")
    hf_model = AutoModelForSequenceClassification.from_pretrained("models/DistilBERT").to(device).eval()

    def get_probs(text):
        enc = hf_tok(text, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            return torch.softmax(hf_model(**enc).logits, dim=-1)[0].cpu().numpy()
else:
    import keras
    from keras.utils import pad_sequences
    model = keras.models.load_model(f"models/{BEST}.keras")
    tokenizer = WordTokenizer.load("models/tokenizer.json")

    def get_probs(text):
        seq = tokenizer.texts_to_sequences([clean_text(text)])
        x = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="post")
        return model.predict(x, verbose=0)[0]


def classify(text):
    if not text or not text.strip():
        return {}, "Please enter a complaint."
    probs = get_probs(text)
    i = int(np.argmax(probs))
    label = CLASSES[i].replace("_", " ").title()
    return ({c: float(p) for c, p in zip(CLASSES, probs)},
            f"**Predicted category:** {label}  \n**Confidence:** {probs[i]:.1%}")


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=8, label="Customer complaint", placeholder="Type or paste the complaint here..."),
    outputs=[gr.Label(num_top_classes=len(CLASSES), label="Category probabilities"), gr.Markdown()],
    title="Consumer Complaint Classifier",
    description=f"Routes a written complaint to the right category. Model: {BEST}",
    examples=[
        ["There is an account on my credit report that is not mine and the bureau refuses to remove it."],
        ["A collection agency keeps calling me about a debt I already paid off."],
        ["My mortgage servicer charged late fees even though my payment was on time."],
        ["I was charged interest on my credit card for a purchase I disputed."],
        ["The bank closed my checking account without notice and is holding my money."],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
