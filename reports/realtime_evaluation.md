# Real-Time Evaluation Report

This report documents the live real-time evaluation of the trained intrusion detection models under emulated network traffic conditions, conforming to the requirements of **Section 12** of the guidelines.

---

## 1. Experimental Setup & Sniffing Interface
The real-time adapter sniffer [src/realtime_adapter.py](file:///d:/HTTP%20flood%20attack/src/realtime_adapter.py) was bound to the loopback interface on Windows:
*   **Capture Driver**: Npcap (installed in WinPcap API-compatible mode).
*   **Interface Name**: `Software Loopback Interface 1`
*   **Protocol Layer**: Transport Layer (Layer 4) packet parsing using Scapy.

---

## 2. Threat Option Fingerprint Discovery
During live testing, we identified that the trained models place high significance on transport-layer fingerprint statistics rather than simple volume count boundaries. Specifically:
1.  **`Fwd Seg Size Min` (TCP Header Option Length)**: Importance = **36.43%**
2.  **`Init Fwd Win Byts` (TCP Client Window Size)**: Importance = **32.54%**
3.  **`Dst Port` (Destination Port)**: Importance = **16.91%**

### Parameter Mapping
*   **Benign Browser Fingerprint**: Standard OS client parameters (Window Size = `8192` or `65535`, standard TCP header = `20` bytes).
*   **Malicious Tool Fingerprint (GoldenEye/Slowloris)**: Hardcoded socket parameters (Window Size = `26883` or `32738`, TCP Options segment size = `32` bytes).

---

## 3. Real-Time Detection Results

### Experiment A: Standard Raw Packets (Benign-Lookalike)
*   **Command**: `python src/emulate_attacks.py --type syn --target 127.0.0.1 --count 500`
*   **Observations**: The packets were sent using Scapy defaults (Window Size = `8192`, TCP Header = `20` bytes).
*   **Prediction**: Classified as **Benign (Class 0)** with **99.8% confidence** because the packet parameters resembled standard client background noise.

### Experiment B: Parameter-Matched Attack (Volumetric Flood)
*   **Command**: 
    ```powershell
    python -c "from scapy.all import IP, TCP, send; send(IP(src='127.0.0.1', dst='127.0.0.1')/TCP(sport=54321, dport=80, flags='S', window=26883, options=[('MSS', 1460), ('NOP', None), ('WScale', 8)]), count=500)"
    ```
*   **Observations**: Packets matched the exact TCP window fingerprint (`26883`) and header options.
*   **Prediction**: Instantly classified as **Attack (Class 1)** with **97.80% confidence** as soon as the flow timed out.

---

## 4. Live Alert Log Output
Below is the captured alert log written to [results/live_alerts.log](file:///d:/HTTP%20flood%20attack/results/live_alerts.log) during Experiment B:

```text
[2026-07-03 16:30:15] 
[ALERT] !!! HTTP FLOOD THREAT DETECTED! !!!
---------------------------------------------
* Flow Key:       127.0.0.1:54321 -> 127.0.0.1:80 (TCP)
* Packets Count:  Fwd=500, Bwd=0
* Confidence:     97.80%
* Classifier:     Catboost
---------------------------------------------
```

---

## 5. Evaluation Conclusions
*   **Zero-Latency Inference**: The real-time adapter successfully processes packet streams, computes flow-based statistics, and queries the pipeline in less than **1 ms** after flow timeout.
*   **Evasion Vector Identified**: The evaluation reveals that signature parameters (like `Init Fwd Win Byts` and `Fwd Seg Size Min`) are heavily leveraged by the model. An attacker using custom client emulation to spoof standard Windows/Chrome TCP window parameters might evade detection, highlighting the need for combined rate-based and payload-based detection boundaries.
