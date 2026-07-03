import sys
import time
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure src in path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import MODELS_DIR
from realtime_adapter import RealtimeFlow

try:
    from scapy.all import IP, TCP, send, sniff
except ImportError:
    print("[ERROR] Scapy is required for this demo. Run 'pip install scapy'")
    sys.exit(1)

def main():
    print("="*70)
    print("DEMONSTRATION: REAL-TIME TRAFFIC COLLECTION & INFERENCE TRACE")
    print("="*70)

    # 1. Load Model and Metadata
    print("\n[STEP 1] Loading CatBoost model and metadata checkpoints...")
    model_path = MODELS_DIR / "final_model.joblib"
    metadata_path = MODELS_DIR / "final_model_metadata.json"
    
    if not model_path.exists():
        print(f"[ERROR] Trained model not found at {model_path}. Please run the pipeline first.")
        sys.exit(1)
        
    model = joblib.load(model_path)
    with open(metadata_path) as f:
        meta = json.load(f)
    feature_names = meta["feature_columns"]
    print(f"  -> Model successfully loaded: {meta['best_model']}")
    print(f"  -> Expecting feature vector size: {len(feature_names)} features")

    # 2. Collect Live Packets
    print("\n[STEP 2] Sniffing live loopback packet streams (Software Loopback Interface 1)...")
    print("  -> Waiting for traffic... (Sending 500 mock attack packets to target port 80 to capture)")
    
    captured_packets = []
    def packet_callback(pkt):
        if pkt.haslayer(IP) and pkt[IP].dst == "127.0.0.1":
            captured_packets.append(pkt)

    # Send packets in a background loop or send them directly to local interface
    send_pkt = IP(src="127.0.0.1", dst="127.0.0.1") / TCP(sport=54321, dport=80, flags="S", window=26883, options=[('MSS', 1460), ('NOP', None), ('WScale', 8)])
    
    # We sniff while sending 500 packets
    # Scapy's send function is called, while sniff captures them
    # To run concurrently, we send 500 packets in a single list
    pkts_to_send = [send_pkt] * 500
    
    start_sniff = time.time()
    send(pkts_to_send, verbose=0)
    
    # Capture loopback packets
    sniff(iface="Software Loopback Interface 1", prn=packet_callback, timeout=2.0)
    
    print(f"  -> Sniffing completed. Captured {len(captured_packets)} raw packets on loopback interface.")
    if len(captured_packets) == 0:
        print("[WARNING] No loopback packets captured. Verify Npcap is installed. Using mock packet array for demo trace.")
        # Fallback to mock packet array
        class MockPacket:
            def __init__(self):
                self.time = time.time()
                self.len = 54
            def haslayer(self, layer):
                return True
        captured_packets = [MockPacket() for _ in range(500)]

    # 3. Flow State Aggregation
    print("\n[STEP 3] Aggregating raw packets into bidirectional flow structure...")
    # Initialize flow key: (src, sport, dst, dport, proto)
    flow_key = ("127.0.0.1", 54321, "127.0.0.1", 80, 6)
    
    # Initialize flow with first packet
    flow = RealtimeFlow(flow_key, captured_packets[0])
    
    # Feed remaining packets
    for pkt in captured_packets[1:]:
        flow.add_packet(pkt, is_fwd=True)
        
    print(f"  -> Flow record created: {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port}")
    print(f"  -> Accumulated packet statistics: Forward Packets = {len(flow.fwd_pkts)}, Backward Packets = {len(flow.bwd_pkts)}")

    # 4. Extract 78-Dimensional Feature Matrix
    print("\n[STEP 4] Extracting 78-dimensional feature matrix for model input...")
    x_flow = flow.extract_features(feature_names)
    
    # Print a few key features for debugging
    print("  -> Extracted features trace (Subset):")
    debug_features = [
        "Dst Port", "Flow Duration", "Tot Fwd Pkts", "TotLen Fwd Pkts", 
        "Fwd Seg Size Min", "Init Fwd Win Byts", "Flow Pkts/s"
    ]
    for feat in debug_features:
        print(f"     * {feat:25s}: {x_flow.loc[0, feat]}")

    # 5. Live Inference and Prediction
    print("\n[STEP 5] Feeding feature matrix to CatBoost model pipeline...")
    pred = model.predict(x_flow)[0]
    probas = model.predict_proba(x_flow)[0]
    confidence = probas[pred]
    label = "Attack (Malicious Flood)" if pred == 1 else "Benign (Legitimate Traffic)"

    print("\n" + "="*70)
    print("DETECTION INFERENCE RESULT")
    print("="*70)
    print(f"* Predicted Label: {label}")
    print(f"* Numeric Output:  {pred}")
    print(f"* Confidence:      {confidence:.2%}")
    print("="*70)

if __name__ == "__main__":
    main()
