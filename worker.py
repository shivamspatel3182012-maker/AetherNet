# Save as: worker.py
import io
import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Automated Hardware Diagnostics Configuration ---
def detect_system_gpu_specs():
    """Queries active hardware drivers to auto-detect GPU models and compute capabilities."""
    if torch.cuda.is_available():
        # Captures the official brand name string directly from the graphics card driver
        gpu_name = torch.cuda.get_device_name(0)
        
        # Pulls the CUDA compute version to calculate rough relative processing power tiers
        major, minor = torch.cuda.get_device_capability(0)
        cuda_arch = f"sm_{major}{minor}"
        
        # Estimate TFLOPS rating based on known hardware generation standards
        # (Allows the master ledger to roughly scale target training epochs)
        if "4090" in gpu_name: tflops = 82.5
        elif "4080" in gpu_name: tflops = 48.7
        elif "4070" in gpu_name: tflops = 29.0
        elif "3090" in gpu_name: tflops = 35.6
        elif "3080" in gpu_name: tflops = 29.8
        else: tflops = 15.0 + (major * 2) # Algorithmic baseline fallback for older architectures
        
        hardware_string = f"NVIDIA_{gpu_name.replace(' ', '_')}_{cuda_arch}"
        return hardware_string, float(tflops)
    else:
        # Fallback parameters if a tester launches the script on an unaccelerated system configuration
        return "CPU_Generic_Host_Node", 2.0

# Execute auto-detection instantly at script runtime initialization
HARDWARE_TYPE, TFLOPS_RATING = detect_system_gpu_specs()
print(f"🖥️ Hardware Diagnostics Complete: Detected {HARDWARE_TYPE} ({TFLOPS_RATING} Estimated TFLOPS)")

class DynamicNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.fc(x)

def run_mining_cycle(tunnel_url, username, local_export_file):
    network_headers = {"Connection": "close"}
    
    # 1. Pull down temporary structural tensor blocks from host node memory pipeline
    try:
        task_res = requests.get(f"{tunnel_url}/task/request", headers=network_headers, timeout=20).json()
        input_dim = task_res["input_dim"]
        target_epochs = task_res["target_epochs"]
        
        # Stream elements directly into processing memory variables
        local_x = torch.tensor(task_res["features"], dtype=torch.float32)
        local_y = torch.tensor(task_res["labels"], dtype=torch.float32)
    except Exception as e:
        print(f"❌ Secure Streaming Link Offline: {e}")
        return False

    # 2. Register with Host Node
    reg_url = f"{tunnel_url}/user/register?username={username}&hardware_type={HARDWARE_TYPE}&tflops={TFLOPS_RATING}"
    try:
        requests.post(reg_url, headers=network_headers, timeout=10)
    except Exception as e:
        print(f"❌ Registration rejected: {e}")
        return False

    # 3. Download the current base model weights
    try:
        model_res = requests.get(f"{tunnel_url}/model/download", headers=network_headers, timeout=15)
        model_res.raise_for_status()
        base_weights = torch.load(io.BytesIO(model_res.content), map_location="cpu")
    except Exception as e:
        print(f"❌ Base Model Configuration Pull Failed: {e}")
        return False

    # 4. Instantiate model layers inside RAM matching server database properties
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicNet(input_dim=input_dim).to(device)
    model.load_state_dict(base_weights)

    dataset = TensorDataset(local_x.to(device), local_y.to(device))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 5. Core AI Training Loop (Adam Optimizer for stability)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🚀 Job Running: Processing {target_epochs} temporary corporate network stream batches [{local_x.shape[0]} rows]...")
    start_time = time.time()
    model.train()
    
    for epoch in range(target_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"   ↳ Step {epoch+1}/{target_epochs} verified. System Loss: {epoch_loss / len(loader):.4f}", flush=True)
            
    seconds_worked = time.time() - start_time
    print(f"🏁 Computations calculated securely in {seconds_worked:.2f}s.")

    # 6. Serialize weight alterations directly to memory or a temp file profile
    model.to("cpu")  
    torch.save(model.state_dict(), local_export_file)

    # 7. Deliver finalized model weight metrics back to network center
    cashout_url = f"{tunnel_url}/user/cashout?username={username}&seconds_worked={seconds_worked}"
    cycle_success = False
    
    try:
        print("📤 Delivering trained architecture adjustments to master vault...", flush=True)
        with open(local_export_file, 'rb') as f:
            files = {'file': (local_export_file, f, 'application/octet-stream')}
            upload_res = requests.post(cashout_url, files=files, headers=network_headers, timeout=30)
            
        if upload_res.status_code == 200:
            print(f"🎉 Success! Proof-of-work locked. Points logged.", flush=True)
            cycle_success = True
        else:
            print(f"❌ Upload rejected by server [{upload_res.status_code}]: {upload_res.text}", flush=True)
    except Exception as e:
        print(f"❌ Upload link blocked: {e}", flush=True)

    # 8. Complete system purge (No logs or raw file elements can linger)
    if os.path.exists(local_export_file):
        os.remove(local_export_file)
        
    # Free reference values from local RAM
    del local_x, local_y, loader, dataset, model
    return cycle_success

def main_execution_loop():
    print("══ Distributed AI Network Private Client Miner ══")
    tunnel_url = input("🔗 Enter Network Tunnel URL (e.g., https://trycloudflare.com): ").strip()
    username = input("👤 Enter your anonymous tracking ID: ").strip()
    
    if tunnel_url.endswith("/"): tunnel_url = tunnel_url[:-1]
    if not tunnel_url or not username:
        print("❌ Inputs mandatory.")
        return

    local_export_file = f"{username}_trained_output.pt"
    successful_jobs = 0
    failed_attempts = 0
    
    print("\n⚡ System initialized. Data streaming activated cleanly over RAM memory.")
    print("🤖 Processing server tasks. Press Ctrl+C to disconnect safely.\n")
    
    while True:
        print(f"═ Pipeline Block Active [Successes: {successful_jobs} | Fails: {failed_attempts}] ═")
        success = run_mining_cycle(tunnel_url, username, local_export_file)
        
        if success:
            successful_jobs += 1
            print("💤 Worker cooling down. Readying for next network pipeline stream...")
            for i in range(15, 0, -1):
                print(f"\r   ↳ Polling next task sequence loop in {i}s... ", end="", flush=True)
                time.sleep(1)
            print("\r   ↳ Pinging network stream master now!             ")
        else:
            failed_attempts += 1
            print("⚠️ Pipeline encounter error. Retrying network configuration link...")
            time.sleep(10)
        print("\n" + "═"*70 + "\n")

if __name__ == "__main__":
    try: main_execution_loop()
    except KeyboardInterrupt: print("\n🛑 Worker dropped out cleanly. Workspace safe.")
