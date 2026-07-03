import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from threading import Thread

# Ensure src in path
sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR, RESULTS_DIR, ID_TO_LABEL
from data_utils import read_json

try:
    from scapy.all import sniff, IP, TCP, UDP
except ImportError:
    print("[ERROR] Scapy is required to run the real-time adapter. Run 'pip install scapy'")
    sys.exit(1)

# Target log for live alerts
LIVE_ALERTS_LOG = RESULTS_DIR / "live_alerts.log"

class RealtimeFlow:
    """Tracks and calculates bidirectional network flow features in real-time."""
    def __init__(self, key, first_pkt):
        self.key = key  # (src_ip, src_port, dst_ip, dst_port, proto)
        self.src_ip, self.src_port, self.dst_ip, self.dst_port, self.proto = key
        
        self.first_time = first_pkt.time
        self.last_time = first_pkt.time
        
        self.fwd_pkts = []
        self.bwd_pkts = []
        
        self.add_packet(first_pkt, is_fwd=True)

    def add_packet(self, pkt, is_fwd):
        pkt_time = pkt.time
        self.last_time = pkt_time
        pkt_len = len(pkt)
        
        # Parse TCP flags if applicable
        flags = ""
        if pkt.haslayer(TCP):
            flags = str(pkt[TCP].flags)

        pkt_info = {
            "time": pkt_time,
            "len": pkt_len,
            "flags": flags,
            "win": pkt[TCP].window if pkt.haslayer(TCP) else 0,
            "header_len": len(pkt[TCP]) if pkt.haslayer(TCP) else len(pkt[UDP]) if pkt.haslayer(UDP) else 0
        }

        if is_fwd:
            self.fwd_pkts.append(pkt_info)
        else:
            self.bwd_pkts.append(pkt_info)

    def extract_features(self, all_feature_names):
        """Computes the exact 78 features required by the final model."""
        f_lens = [p["len"] for p in self.fwd_pkts]
        b_lens = [p["len"] for p in self.bwd_pkts]
        all_lens = f_lens + b_lens

        f_times = [p["time"] for p in self.fwd_pkts]
        b_times = [p["time"] for p in self.bwd_pkts]

        # 1. Flow Duration & Volumes
        flow_duration = max(1, int((self.last_time - self.first_time) * 1e6)) # in microseconds
        tot_fwd_pkts = len(self.fwd_pkts)
        tot_bwd_pkts = len(self.bwd_pkts)
        tot_len_fwd = sum(f_lens)
        tot_len_bwd = sum(b_lens)

        # 2. Inter-Arrival Times (IAT)
        def calc_iat_stats(times):
            if len(times) < 2:
                return 0.0, 0.0, 0.0, 0.0
            iats = np.diff(sorted(times)) * 1e6
            return float(np.mean(iats)), float(np.std(iats)), float(np.max(iats)), float(np.min(iats))

        flow_times = sorted(f_times + b_times)
        f_iat_mean, f_iat_std, f_iat_max, f_iat_min = calc_iat_stats(f_times)
        b_iat_mean, b_iat_std, b_iat_max, b_iat_min = calc_iat_stats(b_times)
        flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = calc_iat_stats(flow_times)

        # 3. Flags Count
        all_flags = "".join([p["flags"] for p in self.fwd_pkts + self.bwd_pkts])
        flag_cnts = {
            "FIN": all_flags.count("F"),
            "SYN": all_flags.count("S"),
            "RST": all_flags.count("R"),
            "PSH": all_flags.count("P"),
            "ACK": all_flags.count("A"),
            "URG": all_flags.count("U"),
            "ECE": all_flags.count("E"),
            "CWE": all_flags.count("C")
        }

        # Header lengths
        fwd_header_len = sum([p["header_len"] for p in self.fwd_pkts])
        bwd_header_len = sum([p["header_len"] for p in self.bwd_pkts])

        # Feature dictionary matching metadata columns
        feat_dict = {
            "Dst Port": self.dst_port,
            "Flow Duration": flow_duration,
            "Tot Fwd Pkts": tot_fwd_pkts,
            "Tot Bwd Pkts": tot_bwd_pkts,
            "TotLen Fwd Pkts": tot_len_fwd,
            "TotLen Bwd Pkts": tot_len_bwd,
            
            "Fwd Pkt Len Max": float(np.max(f_lens)) if f_lens else 0.0,
            "Fwd Pkt Len Min": float(np.min(f_lens)) if f_lens else 0.0,
            "Fwd Pkt Len Mean": float(np.mean(f_lens)) if f_lens else 0.0,
            "Fwd Pkt Len Std": float(np.std(f_lens)) if len(f_lens) > 1 else 0.0,
            
            "Bwd Pkt Len Max": float(np.max(b_lens)) if b_lens else 0.0,
            "Bwd Pkt Len Min": float(np.min(b_lens)) if b_lens else 0.0,
            "Bwd Pkt Len Mean": float(np.mean(b_lens)) if b_lens else 0.0,
            "Bwd Pkt Len Std": float(np.std(b_lens)) if len(b_lens) > 1 else 0.0,
            
            "Flow Byts/s": (tot_len_fwd + tot_len_bwd) / (flow_duration / 1e6),
            "Flow Pkts/s": (tot_fwd_pkts + tot_bwd_pkts) / (flow_duration / 1e6),
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Std": flow_iat_std,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Min": flow_iat_min,
            
            "Fwd IAT Tot": float((f_times[-1] - f_times[0]) * 1e6) if len(f_times) > 1 else 0.0,
            "Fwd IAT Mean": f_iat_mean,
            "Fwd IAT Std": f_iat_std,
            "Fwd IAT Max": f_iat_max,
            "Fwd IAT Min": f_iat_min,
            
            "Bwd IAT Tot": float((b_times[-1] - b_times[0]) * 1e6) if len(b_times) > 1 else 0.0,
            "Bwd IAT Mean": b_iat_mean,
            "Bwd IAT Std": b_iat_std,
            "Bwd IAT Max": b_iat_max,
            "Bwd IAT Min": b_iat_min,
            
            "Fwd PSH Flags": int(any("P" in p["flags"] for p in self.fwd_pkts)),
            "Bwd PSH Flags": int(any("P" in p["flags"] for p in self.bwd_pkts)),
            "Fwd URG Flags": int(any("U" in p["flags"] for p in self.fwd_pkts)),
            "Bwd URG Flags": int(any("U" in p["flags"] for p in self.bwd_pkts)),
            
            "Fwd Header Len": fwd_header_len,
            "Bwd Header Len": bwd_header_len,
            "Fwd Pkts/s": tot_fwd_pkts / (flow_duration / 1e6),
            "Bwd Pkts/s": tot_bwd_pkts / (flow_duration / 1e6),
            
            "Pkt Len Min": float(np.min(all_lens)) if all_lens else 0.0,
            "Pkt Len Max": float(np.max(all_lens)) if all_lens else 0.0,
            "Pkt Len Mean": float(np.mean(all_lens)) if all_lens else 0.0,
            "Pkt Len Std": float(np.std(all_lens)) if len(all_lens) > 1 else 0.0,
            "Pkt Len Var": float(np.var(all_lens)) if len(all_lens) > 1 else 0.0,
            
            "FIN Flag Cnt": flag_cnts["FIN"],
            "SYN Flag Cnt": flag_cnts["SYN"],
            "RST Flag Cnt": flag_cnts["RST"],
            "PSH Flag Cnt": flag_cnts["PSH"],
            "ACK Flag Cnt": flag_cnts["ACK"],
            "URG Flag Cnt": flag_cnts["URG"],
            "CWE Flag Count": flag_cnts["CWE"],
            "ECE Flag Cnt": flag_cnts["ECE"],
            
            "Down/Up Ratio": tot_bwd_pkts / tot_fwd_pkts if tot_fwd_pkts > 0 else 0.0,
            "Pkt Size Avg": float(np.mean(all_lens)) if all_lens else 0.0,
            "Fwd Seg Size Avg": float(np.mean(f_lens)) if f_lens else 0.0,
            "Bwd Seg Size Avg": float(np.mean(b_lens)) if b_lens else 0.0,
            
            "Fwd Byts/b Avg": 0.0, "Fwd Pkts/b Avg": 0.0, "Fwd Blk Rate Avg": 0.0,
            "Bwd Byts/b Avg": 0.0, "Bwd Pkts/b Avg": 0.0, "Bwd Blk Rate Avg": 0.0,
            
            "Subflow Fwd Pkts": tot_fwd_pkts,
            "Subflow Fwd Byts": tot_len_fwd,
            "Subflow Bwd Pkts": tot_bwd_pkts,
            "Subflow Bwd Byts": tot_len_bwd,
            
            "Init Fwd Win Byts": self.fwd_pkts[0]["win"] if self.fwd_pkts else 0,
            "Init Bwd Win Byts": self.bwd_pkts[0]["win"] if self.bwd_pkts else 0,
            "Fwd Act Data Pkts": sum(1 for p in self.fwd_pkts if p["len"] > p["header_len"]),
            "Fwd Seg Size Min": min([p["header_len"] for p in self.fwd_pkts]) if self.fwd_pkts else 0,
            
            "Active Mean": 0.0, "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0
        }

        # Return values aligned exactly to metadata features order as a Pandas DataFrame
        ordered_vals = [feat_dict.get(feat, 0.0) for feat in all_feature_names]
        return pd.DataFrame([ordered_vals], columns=all_feature_names)


class LiveDetector:
    """Manages active network sniffing and scores traffic using the final model."""
    def __init__(self, model_path, metadata_path, interface=None):
        self.interface = interface
        self.model = joblib.load(model_path)
        self.metadata = read_json(metadata_path)
        self.feature_names = self.metadata["feature_columns"]
        
        self.active_flows = {}
        self.flow_timeout = 6.0  # seconds

    def process_packet(self, pkt):
        if not pkt.haslayer(IP):
            return

        ip_layer = pkt[IP]
        proto = ip_layer.proto
        
        # Filter TCP & UDP
        if proto not in [6, 17]:
            return

        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        else:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        # Define 5-tuple
        fwd_key = (src_ip, src_port, dst_ip, dst_port, proto)
        bwd_key = (dst_ip, dst_port, src_ip, src_port, proto)

        if fwd_key in self.active_flows:
            self.active_flows[fwd_key].add_packet(pkt, is_fwd=True)
        elif bwd_key in self.active_flows:
            self.active_flows[bwd_key].add_packet(pkt, is_fwd=False)
        else:
            # Create new flow
            self.active_flows[fwd_key] = RealtimeFlow(fwd_key, pkt)

    def flush_expired_flows(self):
        """Periodically scans active flows, scores them, and flushes expired ones."""
        while True:
            time.sleep(2.0)
            now = time.time()
            expired_keys = []

            for key, flow in list(self.active_flows.items()):
                if now - flow.last_time > self.flow_timeout:
                    expired_keys.append(key)
                    
                    # Compute feature matrix
                    x_flow = flow.extract_features(self.feature_names)
                    
                    # Inference
                    pred = self.model.predict(x_flow)[0]
                    probas = self.model.predict_proba(x_flow)[0]
                    confidence = probas[pred]
                    
                    if pred == 1:
                        # Threat Alert
                        alert_msg = (
                            f"\n[ALERT] 🚨 HTTP FLOOD THREAT DETECTED! 🚨\n"
                            f"---------------------------------------------\n"
                            f"* Flow Key:       {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} ({'TCP' if flow.proto==6 else 'UDP'})\n"
                            f"* Packets Count:  Fwd={len(flow.fwd_pkts)}, Bwd={len(flow.bwd_pkts)}\n"
                            f"* Confidence:     {confidence:.2%}\n"
                            f"* Classifier:     {self.metadata['best_model'].title()}\n"
                            f"---------------------------------------------\n"
                        )
                        print(alert_msg)
                        with open(LIVE_ALERTS_LOG, "a", encoding="utf-8") as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {alert_msg}")
                    else:
                        print(f"  [PASS] Clean flow: {flow.src_ip} -> {flow.dst_ip}:{flow.dst_port}")

            for k in expired_keys:
                if k in self.active_flows:
                    del self.active_flows[k]

    def start_sniffing(self):
        print("="*60)
        print("REAL-TIME DETECTION ADAPTER INITIALIZED")
        print("="*60)
        print(f"Loaded classifier: {self.metadata['best_model']}")
        print(f"Monitoring feature columns: {len(self.feature_names)}")
        print(f"Listening on interface: {self.interface or 'Default Interface'}")
        print("Monitoring active flow states... (Press Ctrl+C to terminate)")
        print("="*60)

        # Launch flush thread
        flush_thread = Thread(target=self.flush_expired_flows, daemon=True)
        flush_thread.start()

        # Start sniffer
        sniff_kwargs = {"prn": self.process_packet, "store": False}
        if self.interface:
            sniff_kwargs["iface"] = self.interface

        sniff(**sniff_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Detection Adapter Sniffer")
    parser.add_argument("--interface", type=str, default=None, help="Name of the interface to sniff (e.g. Loopback)")
    args = parser.parse_args()

    model_path = MODELS_DIR / "final_model.joblib"
    metadata_path = MODELS_DIR / "final_model_metadata.json"

    if not model_path.exists():
        print(f"[ERROR] Trained final model binary not found at: {model_path}. Train the pipeline first.")
        sys.exit(1)

    detector = LiveDetector(model_path, metadata_path, interface=args.interface)
    try:
        detector.start_sniffing()
    except KeyboardInterrupt:
        print("\n[INFO] Real-time detection adapter terminated by user.")
        sys.exit(0)
