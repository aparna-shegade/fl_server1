from flask import Flask, request, jsonify
import numpy as np
import json
import threading
import os
import requests

app = Flask(__name__)

# ─────────────────────────────────────────
# Download large files from HuggingFace
# ─────────────────────────────────────────
HF_BASE = "https://huggingface.co/Aiman18/mymodel/resolve/main"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        headers = {}
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        r = requests.get(url, headers=headers, stream=True)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ {filename} downloaded!")
        else:
            print(f"❌ Failed to download {filename} — Status: {r.status_code}")
    else:
        print(f"✅ {filename} already exists, skipping download.")

# Download all large files at startup
download_file(f"{HF_BASE}/initial_weights.json", "initial_weights.json")
download_file(f"{HF_BASE}/fl_model.tflite",       "fl_model.tflite")
download_file(f"{HF_BASE}/vocab.json",             "vocab.json")

# ─────────────────────────────────────────
# Load initial weights (after download)
# ─────────────────────────────────────────
with open("initial_weights.json") as f:
    global_weights = json.load(f)

# ─────────────────────────────────────────
# FL State
# ─────────────────────────────────────────
client_weights = []
MIN_CLIENTS    = 4
lock           = threading.Lock()
round_number   = 0

# ─────────────────────────────────────────
# FedAvg Aggregation
# ─────────────────────────────────────────
def fedavg(weights_list):
    """Average weights across all clients"""
    avg = []
    for layer_idx in range(len(weights_list[0])):
        layer_weights = [w[layer_idx] for w in weights_list]
        avg.append(np.mean(layer_weights, axis=0).tolist())
    return avg

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route("/get_global_model", methods=["GET"])
def get_global_model():
    return jsonify({
        "weights": global_weights,
        "round":   round_number
    })

@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    global global_weights, client_weights, round_number

    data       = request.json
    device_id  = data.get("device_id")
    weights    = data.get("weights")
    num_samples = data.get("num_samples", 1)

    with lock:
        client_weights.append({
            "device_id":   device_id,
            "weights":     weights,
            "num_samples": num_samples
        })
        collected = len(client_weights)
        print(f"📥 Received weights from {device_id} — {collected}/{MIN_CLIENTS}")

        if collected >= MIN_CLIENTS:
            print("🔄 Aggregating weights...")
            all_weights  = [c["weights"] for c in client_weights]
            global_weights = fedavg(all_weights)
            client_weights = []
            round_number  += 1
            print(f"✅ Round {round_number} complete!")
            return jsonify({
                "status":  "aggregated",
                "round":   round_number,
                "message": f"Round {round_number} complete!"
            })

    return jsonify({
        "status":    "waiting",
        "collected": collected,
        "needed":    MIN_CLIENTS
    })

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "round":           round_number,
        "clients_waiting": len(client_weights),
        "min_clients":     MIN_CLIENTS
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ─────────────────────────────────────────
# Entry Point — Render uses dynamic PORT
# ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ← Critical for Render
    app.run(host="0.0.0.0", port=port)
