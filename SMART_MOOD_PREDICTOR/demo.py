# demo.py
import os
from joblib import load, dump
import numpy as np
import pandas as pd
import gradio as gr
from train import generate_synthetic, train_and_save
from sklearn.pipeline import Pipeline
from io import BytesIO
import matplotlib.pyplot as plt

MODEL_PATH = "models/smartmood_model.joblib"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("No model found — training a model now (this takes ~20-60s)...")
        train_and_save()
    return load(MODEL_PATH)

model = ensure_model()

def simulate_day(seed=0):
    rng = np.random.RandomState(seed)
    # Create a single day sample
    screen_time = float(np.clip(rng.normal(240, 90), 10, 1440))
    typing_speed = float(np.clip(rng.normal(200, 70), 10, 1000))
    app_switches = float(rng.poisson(8))
    orientation = float(rng.poisson(15))
    notifications = float(rng.poisson(12))
    return {
        "screen_time": screen_time,
        "typing_speed": typing_speed,
        "app_switches": app_switches,
        "orientation_changes": orientation,
        "notifications": notifications
    }

def predict_from_inputs(screen_time, typing_speed, app_switches, orientation_changes, notifications, trend_days=7, seed=0):
    X = [[screen_time, typing_speed, app_switches, orientation_changes, notifications]]
    pred = model.predict(X)[0]
    probs = dict(zip(model.classes_, model.predict_proba(X)[0].round(4)))
    suggestions = {
        "Happy": "Keep that up! Consider sharing the good mood 😊",
        "Neutral": "Maintain steady habits — try a short walk.",
        "Tired": "You look tired — a short nap or rest might help.",
        "Stressed": "Try breathing exercises, stop for 10 minutes, or listen to calming music."
    }
    suggestion = suggestions.get(pred, "")

    rng = np.random.RandomState(seed)
    days = np.arange(trend_days)
    trend_samples = []
    for d in days:
        samp = simulate_day(seed + d)
        trend_samples.append([
            samp["screen_time"], samp["typing_speed"],
            samp["app_switches"], samp["orientation_changes"],
            samp["notifications"]
        ])
    trend_samples = np.array(trend_samples)
    preds = model.predict(trend_samples)

    # Plot mood trend
    fig, ax = plt.subplots(figsize=(6, 2.8))
    mood_map = {"Stressed": 0, "Tired": 1, "Neutral": 2, "Happy": 3}
    ax.plot(range(1, trend_days + 1), [mood_map[p] for p in preds], marker="o")
    ax.set_yticks(list(mood_map.values()))
    ax.set_yticklabels(list(mood_map.keys()))
    ax.set_xlabel("Day")
    ax.set_title("Simulated mood trend (last {} days)".format(trend_days))
    plt.tight_layout()

    from io import BytesIO
    import PIL.Image as Image
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    return pred, probs, suggestion, img

with gr.Blocks() as demo:
    gr.Markdown("# SmartMood — Real-time emotion from device usage\nPredict mood from screen time, typing speed, app-switching and notifications.\n")
    with gr.Row():
        with gr.Column():
            screen_time = gr.Slider(0, 1440, value=240, label="Screen time (minutes/day)")
            typing_speed = gr.Slider(0, 1000, value=200, label="Typing speed (chars/minute)")
            app_switches = gr.Slider(0, 60, value=8, label="App switches per hour")
            orientation_changes = gr.Slider(0, 120, value=15, label="Orientation changes per hour")
            notifications = gr.Slider(0, 120, value=12, label="Notifications per hour")
            trend_days = gr.Slider(3, 14, value=7, step=1, label="Trend (days to simulate)")
            seed = gr.Number(value=0, visible=False)
            predict_btn = gr.Button("Predict Mood")
        with gr.Column():
            out_label = gr.Label(value="Prediction will appear here")
            out_probs = gr.Dataframe(headers=["label","probability"], datatype=["str","number"], interactive=False)
            out_suggestion = gr.Textbox(label="Suggestion", lines=3)
            out_plot = gr.Image()

    def on_predict(st, tp, ap, oc, nt, td, s):
        pred, probs, suggestion, buf = predict_from_inputs(st, tp, ap, oc, nt, td, int(s))
        prob_pairs = [[k, float(v)] for k, v in probs.items()]
        return pred, prob_pairs, suggestion, buf

    predict_btn.click(fn=on_predict,
                      inputs=[screen_time, typing_speed, app_switches, orientation_changes, notifications, trend_days, seed],
                      outputs=[out_label, out_probs, out_suggestion, out_plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
