from flask import Flask, request, jsonify
import numpy as np
import json
import threading
import copy

app = Flask(__name__)

# Load initial weights
with open("initial_weights.json") as f:
    global_weights = json.load(f)

# FL state
client_weights = []       # collected from devices
MIN_CLIENTS = 4           # wait for all 4 before aggregating
lock = threading.Lock()
round_number = 0

def fedavg(weights_list):
    """Average weights across all clients"""
    avg = []
    for layer_idx in range(len(weights_list[0])):
        layer_weights = [w[layer_idx] for w in weights_list]
        avg.append(np.mean(layer_weights, axis=0).tolist())
    return avg

@app.route("/get_global_model", methods=["GET"])
def get_global_model():
    return jsonify({
        "weights": global_weights,
        "round": round_number
    })

@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    global global_weights, client_weights, round_number

    data = request.json
    device_id = data.get("device_id")
    weights = data.get("weights")
    num_samples = data.get("num_samples", 1)

    with lock:
        client_weights.append({
            "device_id": device_id,
            "weights": weights,
            "num_samples": num_samples
        })
        collected = len(client_weights)
        print(f"Received weights from {device_id} — {collected}/{MIN_CLIENTS}")

        if collected >= MIN_CLIENTS:
            print("Aggregating...")
            all_weights = [c["weights"] for c in client_weights]
            global_weights = fedavg(all_weights)
            client_weights = []
            round_number += 1
            print(f"Round {round_number} complete!")
            return jsonify({
                "status": "aggregated",
                "round": round_number,
                "message": f"Round {round_number} complete!"
            })

    return jsonify({
        "status": "waiting",
        "collected": collected,
        "needed": MIN_CLIENTS
    })

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "round": round_number,
        "clients_waiting": len(client_weights),
        "min_clients": MIN_CLIENTS
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)