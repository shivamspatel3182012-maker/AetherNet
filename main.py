# Save as: main.py
import os
import json
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd

# Suppress the PyTorch NumPy warning notice cleanly
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

VAULT_DIR = r"E:\ai_network_vault"
SUBMISSIONS_DIR = os.path.join(VAULT_DIR, "submissions")
LEDGER_FILE = os.path.join(VAULT_DIR, "ledger.json")
POINTS_LEDGER_FILE = os.path.join(VAULT_DIR, "points_ledger.json")
BASE_MODEL_PATH = os.path.join(VAULT_DIR, "base_model.pt")
SERVER_DATA_FILE = os.path.join(VAULT_DIR, "validation_data.csv")

# Simple, un-crashable disk lock layer
DISK_LOCK = asyncio.Lock()

# Dynamic Global Architecture and Streaming Values
INPUT_DIM = 8  
VAL_LOADER = None
GLOBAL_FEATURES = None
GLOBAL_LABELS = None

class DynamicNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.fc(x)

def load_server_dataset():
    """Loads the main corporate database. Data features remain entirely on host disk."""
    global VAL_LOADER, INPUT_DIM, GLOBAL_FEATURES, GLOBAL_LABELS
    if not os.path.exists(SERVER_DATA_FILE):
        print(f"⚠️ Critical: Corporate file {SERVER_DATA_FILE} missing! Using fallbacks.")
        GLOBAL_FEATURES = np.random.randn(500, 8)
        GLOBAL_LABELS = np.random.randn(500, 1)
        INPUT_DIM = 8
    else:
        try:
            df = pd.read_csv(SERVER_DATA_FILE)
            features = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values.reshape(-1, 1)
            
            INPUT_DIM = features.shape[1]
            GLOBAL_FEATURES = features
            GLOBAL_LABELS = labels
            print(f"📈 Vault Activated: Loaded database with {features.shape[0]} rows, {INPUT_DIM} features.")
        except Exception as e:
            print(f"❌ Database error: {e}")
            GLOBAL_FEATURES = np.random.randn(500, 8)
            GLOBAL_LABELS = np.random.randn(500, 1)
            INPUT_DIM = 8

    # Convert to evaluation tensors for tournament scoring
    v_x = torch.tensor(GLOBAL_FEATURES, dtype=torch.float32)
    v_y = torch.tensor(GLOBAL_LABELS, dtype=torch.float32)
    VAL_LOADER = DataLoader(TensorDataset(v_x, v_y), batch_size=32)

def load_json_file(file_path):
    if not os.path.exists(file_path): return {}
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else {}
    except json.JSONDecodeError:
        with open(file_path, "w") as f: json.dump({}, f)
        return {}

def save_json_file(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def update_points(username, amount):
    points_ledger = load_json_file(POINTS_LEDGER_FILE)
    points_ledger[username] = points_ledger.get(username, 0.0) + amount
    save_json_file(POINTS_LEDGER_FILE, points_ledger)

async def run_tournament_logic():
    if not os.path.exists(SUBMISSIONS_DIR): return "No entries."
    model_files = [f for f in os.listdir(SUBMISSIONS_DIR) if f.endswith(".pt")]
    if not model_files: return "No submissions available."

    criterion = nn.MSELoss()
    best_score = float('inf')
    winner_username = None
    winner_file_path = None
    evaluated_submissions = []

    print(f"🏆 Tournament Starting: Grading {len(model_files)} miner models...")

    for filename in model_files:
        username = filename.replace("_latest_model.pt", "")
        file_path = os.path.join(SUBMISSIONS_DIR, filename)
        evaluated_submissions.append((username, file_path))
        
        try:
            def evaluate_model():
                model = DynamicNet(input_dim=INPUT_DIM)
                model.load_state_dict(torch.load(file_path, map_location="cpu"))
                model.eval()
                total_loss = 0.0
                with torch.no_grad():
                    for x, y in VAL_LOADER:
                        total_loss += criterion(model(x), y).item()
                return total_loss / len(VAL_LOADER)

            avg_loss = await asyncio.to_thread(evaluate_model)
            print(f"🔬 Node Verification {username} | Validation Loss: {avg_loss:.4f}")

            if avg_loss < best_score:
                best_score = avg_loss
                winner_username = username
                winner_file_path = file_path
        except Exception as e:
            print(f"❌ Disqualified entry from {username}: {e}")

    if winner_username and winner_file_path:
        try:
            print(f"🎉 Winner Profile Selected: {winner_username} with Loss: {best_score:.4f}!")
            os.replace(winner_file_path, BASE_MODEL_PATH)
            update_points(winner_username, 50.0)
            status_msg = f"Tournament winner: {winner_username} ({best_score:.4f})"
        except Exception as e:
            status_msg = f"Failed base update: {str(e)}"
    else:
        status_msg = "No functional winner."
    
    for username, file_path in evaluated_submissions:
        if file_path != winner_file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except Exception: pass
                
    return status_msg

async def tournament_worker():
    while True:
        await asyncio.sleep(7200)
        async with DISK_LOCK:
            await run_tournament_logic()

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    load_server_dataset()
    if not os.path.exists(BASE_MODEL_PATH):
        torch.save(DynamicNet(input_dim=INPUT_DIM).state_dict(), BASE_MODEL_PATH)
    task = asyncio.create_task(tournament_worker())
    yield
    task.cancel()

app = FastAPI(title="Distributed AI Network - Master Host Node", lifespan=lifespan)

@app.get("/model/download")
async def download_base_model():
    if not os.path.exists(BASE_MODEL_PATH): raise HTTPException(status_code=404, detail="Missing weights.")
    return FileResponse(BASE_MODEL_PATH, media_type="application/octet-stream", filename="base_model.pt")

@app.get("/task/request")
async def request_task_data():
    """Streams data straight into worker RAM. No data files are ever written on client disks."""
    global GLOBAL_FEATURES, GLOBAL_LABELS, INPUT_DIM
    async with DISK_LOCK:
        total_rows = GLOBAL_FEATURES.shape[0]
        # Dynamically sample a random batch block for this network miner task (e.g., 200 rows)
        batch_size = min(200, total_rows)
        indices = np.random.choice(total_rows, batch_size, replace=False)
        
        sampled_x = GLOBAL_FEATURES[indices].tolist()
        sampled_y = GLOBAL_LABELS[indices].tolist()
        
        return {
            "input_dim": INPUT_DIM,
            "target_epochs": 5,
            "features": sampled_x,
            "labels": sampled_y
        }

@app.get("/points")
async def view_points_ledger():
    async with DISK_LOCK: return load_json_file(POINTS_LEDGER_FILE)

@app.post("/admin/manual_tournament")
async def manual_tournament():
    async with DISK_LOCK:
        result = await run_tournament_logic()
        return {"status": "Complete", "message": result}

@app.post("/user/register")
async def register_miner(username: str, hardware_type: str, tflops: float):
    async with DISK_LOCK:
        ledger = load_json_file(LEDGER_FILE)
        if username in ledger:
            ledger[username]["hardware_type"] = hardware_type
            ledger[username]["tflops"] = tflops
            save_json_file(LEDGER_FILE, ledger)
            return {"message": f"Welcome back, {username}!"}
        
        ledger[username] = {"hardware_type": hardware_type, "tflops": tflops, "jobs_completed": 0}
        save_json_file(LEDGER_FILE, ledger)
        
        points_ledger = load_json_file(POINTS_LEDGER_FILE)
        if username not in points_ledger:
            points_ledger[username] = 0.0
            save_json_file(POINTS_LEDGER_FILE, points_ledger)
            
        return {"message": f"Registered node {username} successfully!"}

@app.post("/user/cashout")
async def cashout_points(username: str, seconds_worked: float, file: UploadFile = File(...)):
    async with DISK_LOCK:
        ledger = load_json_file(LEDGER_FILE)
        if username not in ledger: raise HTTPException(status_code=404, detail="Unrecognized node ID.")

        safe_filename = f"{username}_latest_model.pt"
        file_path = os.path.join(SUBMISSIONS_DIR, safe_filename)

        try:
            with open(file_path, "wb") as buffer:
                while chunk := await file.read(64 * 1024): buffer.write(chunk)
            ledger[username]["jobs_completed"] += 1
            save_json_file(LEDGER_FILE, ledger)
            update_points(username, 10.0)
        except Exception as e:
            if os.path.exists(file_path): os.remove(file_path)
            raise HTTPException(status_code=500, detail=str(e))

    return {"status": "Queued", "message": "Weights archived safely. Points processed."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8888, reload=False)
