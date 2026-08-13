# Save as: worker.py
import os
import time
import random
import threading
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import tkinter as tk
from tkinter import scrolledtext

# ==========================================
# 🎨 STYLING & HUD THEME CONSTANTS
# ==========================================
BG_DARK = "#0d1117"
PANEL_BG = "#161b22"
BORDER_COLOR = "#30363d"
CYAN_GLOW = "#58a6ff"
GREEN_GLOW = "#2ea043"
ACCENT_PURPLE = "#bc8cff"
TEXT_PRIMARY = "#c9d1d9"
TEXT_MUTED = "#8b949e"
ERROR_RED = "#f85149"


# ==========================================
# 🧠 DYNAMIC NET ARCHITECTURE (MATCHES SERVER)
# ==========================================
class DynamicNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


# ==========================================
# 🎛️ WORKER DASHBOARD
# ==========================================
class AthernetWorkerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ATHERNET // WORKER NODE")
        self.root.geometry("880x640")
        self.root.configure(bg=BG_DARK)
        self.is_mining = False

        self.loss_history = []
        self.gauge_val = 0
        self.points_earned = 0

        self._build_header()
        self._build_input_bar()
        self._build_hud_grid()
        self._build_terminal()

        self.animate_hud()

    def _build_header(self):
        header = tk.Frame(self.root, bg=PANEL_BG, bd=1, relief="solid")
        header.pack(fill="x", padx=15, pady=(15, 5), ipady=8)

        title = tk.Label(
            header, 
            text="❖ ATHERNET COMPUTE NODE", 
            font=("Consolas", 14, "bold"), 
            fg=CYAN_GLOW, 
            bg=PANEL_BG
        )
        title.pack(side="left", padx=15)

        self.status_badge = tk.Label(
            header, 
            text="SYSTEM IDLE", 
            font=("Consolas", 10, "bold"), 
            fg=TEXT_MUTED, 
            bg="#21262d", 
            padx=10, 
            pady=3
        )
        self.status_badge.pack(side="right", padx=15)

    def _build_input_bar(self):
        input_frame = tk.Frame(self.root, bg=PANEL_BG, highlightbackground=BORDER_COLOR, highlightthickness=1)
        input_frame.pack(fill="x", padx=15, pady=5, ipady=5)

        tk.Label(input_frame, text="Host URL:", font=("Consolas", 9, "bold"), fg=TEXT_MUTED, bg=PANEL_BG).grid(row=0, column=0, padx=(15, 5), pady=5, sticky="w")
        self.url_entry = tk.Entry(input_frame, width=38, bg="#0d1117", fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, relief="flat", font=("Consolas", 9))
        self.url_entry.insert(0, "http://127.0.0.1:8888")
        self.url_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(input_frame, text="Miner ID:", font=("Consolas", 9, "bold"), fg=TEXT_MUTED, bg=PANEL_BG).grid(row=0, column=2, padx=(15, 5), pady=5, sticky="w")
        self.user_entry = tk.Entry(input_frame, width=18, bg="#0d1117", fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, relief="flat", font=("Consolas", 9))
        self.user_entry.insert(0, "miner_node_01")
        self.user_entry.grid(row=0, column=3, padx=5, pady=5)

    def _build_hud_grid(self):
        grid_frame = tk.Frame(self.root, bg=BG_DARK)
        grid_frame.pack(fill="x", padx=15, pady=5)

        gauge_panel = tk.Frame(grid_frame, bg=PANEL_BG, highlightbackground=BORDER_COLOR, highlightthickness=1)
        gauge_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

        tk.Label(gauge_panel, text="COMPUTE WORKLOAD", font=("Consolas", 9, "bold"), fg=TEXT_MUTED, bg=PANEL_BG).pack(anchor="w", padx=10, pady=(5,0))
        self.gauge_canvas = tk.Canvas(gauge_panel, width=220, height=130, bg=PANEL_BG, highlightthickness=0)
        self.gauge_canvas.pack(pady=5)

        graph_panel = tk.Frame(grid_frame, bg=PANEL_BG, highlightbackground=BORDER_COLOR, highlightthickness=1)
        graph_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

        tk.Label(graph_panel, text="TRAINING LOSS TELEMETRY", font=("Consolas", 9, "bold"), fg=TEXT_MUTED, bg=PANEL_BG).pack(anchor="w", padx=10, pady=(5,0))
        self.graph_canvas = tk.Canvas(graph_panel, width=570, height=130, bg=PANEL_BG, highlightthickness=0)
        self.graph_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=2)

    def _build_terminal(self):
        ctrl_frame = tk.Frame(self.root, bg=BG_DARK)
        ctrl_frame.pack(fill="x", padx=15, pady=5)

        self.btn_toggle = tk.Button(
            ctrl_frame, 
            text="▶ START MINING", 
            font=("Consolas", 10, "bold"), 
            fg="#111", 
            bg=GREEN_GLOW, 
            activebackground=CYAN_GLOW, 
            command=self.toggle_mining, 
            relief="flat", 
            padx=15, 
            pady=5
        )
        self.btn_toggle.pack(side="left")

        self.points_label = tk.Label(ctrl_frame, text="EARNED: 0 PTS", font=("Consolas", 11, "bold"), fg=ACCENT_PURPLE, bg=BG_DARK)
        self.points_label.pack(side="right", padx=10)

        log_frame = tk.Frame(self.root, bg=PANEL_BG, highlightbackground=BORDER_COLOR, highlightthickness=1)
        log_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        self.console = scrolledtext.ScrolledText(
            log_frame, 
            bg="#0d1117", 
            fg=TEXT_PRIMARY, 
            insertbackground=TEXT_PRIMARY, 
            font=("Consolas", 9), 
            relief="flat"
        )
        self.console.pack(fill="both", expand=True, padx=5, pady=5)

    def draw_gauge(self, value):
        self.gauge_canvas.delete("all")
        cx, cy, r = 110, 100, 70
        self.gauge_canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=-30, extent=240, style="arc", outline="#21262d", width=10)
        extent = (value / 100.0) * 240
        gauge_color = GREEN_GLOW if value < 80 else ERROR_RED
        self.gauge_canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=210, extent=-extent, style="arc", outline=gauge_color, width=10)
        self.gauge_canvas.create_text(cx, cy-5, text=f"{int(value)}%", font=("Consolas", 16, "bold"), fill=TEXT_PRIMARY)
        self.gauge_canvas.create_text(cx, cy+18, text="GPU/CPU LOAD", font=("Consolas", 8), fill=TEXT_MUTED)

    def draw_graph(self):
        self.graph_canvas.delete("all")
        w, h = 550, 110
        pad = 20

        for i in range(1, 4):
            y = pad + i * ((h - 2 * pad) / 4)
            self.graph_canvas.create_line(pad, y, w - pad, y, fill="#21262d", dash=(2, 4))

        if len(self.loss_history) < 2:
            return

        max_val = max(self.loss_history) if max(self.loss_history) > 0 else 1.0
        min_val = min(self.loss_history)
        points = []
        x_step = (w - 2 * pad) / (len(self.loss_history) - 1)

        for idx, val in enumerate(self.loss_history):
            x = pad + idx * x_step
            y = (h - pad) - ((val - min_val) / (max_val - min_val + 1e-5)) * (h - 2 * pad)
            points.append((x, y))

        for i in range(len(points) - 1):
            self.graph_canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1], fill=CYAN_GLOW, width=2)
            self.graph_canvas.create_oval(points[i][0]-3, points[i][1]-3, points[i][0]+3, points[i][1]+3, fill=ACCENT_PURPLE, outline="")

    def animate_hud(self):
        if self.is_mining:
            self.gauge_val = min(100, max(45, self.gauge_val + random.randint(-10, 10)))
        else:
            self.gauge_val = max(0, self.gauge_val - 5)

        self.draw_gauge(self.gauge_val)
        self.draw_graph()
        self.root.after(100, self.animate_hud)

    def log(self, text):
        def _append():
            self.console.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
            self.console.see(tk.END)
        self.root.after(0, _append)

    def toggle_mining(self):
        if not self.is_mining:
            url = self.url_entry.get().strip().rstrip("/")
            username = self.user_entry.get().strip()

            if not url or not username:
                self.log("⚠️ Error: Please enter a valid Host URL and Miner ID.")
                return

            self.is_mining = True
            self.gauge_val = 60
            self.btn_toggle.config(text="⏹ STOP MINING", bg=ERROR_RED, fg="#fff")
            self.status_badge.config(text="MINING ACTIVE", fg="#fff", bg=GREEN_GLOW)
            self.url_entry.config(state="disabled")
            self.user_entry.config(state="disabled")
            
            threading.Thread(target=self.run_worker_pipeline, args=(url, username), daemon=True).start()
        else:
            self.is_mining = False
            self.btn_toggle.config(text="▶ START MINING", bg=GREEN_GLOW, fg="#111")
            self.status_badge.config(text="SYSTEM IDLE", fg=TEXT_MUTED, bg="#21262d")
            self.url_entry.config(state="normal")
            self.user_entry.config(state="normal")
            self.log("⏸️ Stop request issued. Shutting down worker...")

    def run_worker_pipeline(self, host_url, miner_id):
        self.log(f"🚀 Initializing compute link to {host_url} as [{miner_id}]...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"💻 Hardware acceleration: [{device.type.upper()}]")

        while self.is_mining:
            try:
                # 1. Fetch Task Request from Host Node
                req_url = f"{host_url}/task/request"
                resp = requests.get(req_url, timeout=10)

                if resp.status_code != 200:
                    self.log(f"⚠️ Server response code: {resp.status_code}. Retrying in 5s...")
                    time.sleep(5)
                    continue

                data = resp.json()
                features = np.array(data["features"], dtype=np.float32)
                labels = np.array(data["labels"], dtype=np.float32)
                input_dim = data.get("input_dim", features.shape[1])

                inputs = torch.tensor(features).to(device)
                targets = torch.tensor(labels).to(device).unsqueeze(1) if labels.ndim == 1 else torch.tensor(labels).to(device)

                # 2. Local Model Optimization (Using exact DynamicNet architecture)
                dataset = TensorDataset(inputs, targets)
                loader = DataLoader(dataset, batch_size=32, shuffle=True)

                model = DynamicNet(input_dim=input_dim).to(device)
                
                # Optionally attempt base model weights download
                try:
                    dl_resp = requests.get(f"{host_url}/model/download", timeout=5)
                    if dl_resp.status_code == 200:
                        temp_dl_path = f"temp_base_{miner_id}.pt"
                        with open(temp_dl_path, "wb") as f:
                            f.write(dl_resp.content)
                        model.load_state_dict(torch.load(temp_dl_path, map_location=device, weights_only=True))
                        if os.path.exists(temp_dl_path):
                            os.remove(temp_dl_path)
                except Exception:
                    pass # Fallback to local weights if base download fails

                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.01)

                model.train()
                final_loss = 0.0
                for epoch in range(5):
                    if not self.is_mining:
                        break
                    for batch_x, batch_y in loader:
                        optimizer.zero_grad()
                        preds = model(batch_x)
                        loss = criterion(preds, batch_y)
                        loss.backward()
                        optimizer.step()
                        final_loss = loss.item()

                if not self.is_mining:
                    break

                rounded_loss = round(final_loss, 4)
                self.loss_history.append(rounded_loss)
                if len(self.loss_history) > 18:
                    self.loss_history.pop(0)

                # 3. Serializing & Posting Model Binary to /user/cashout
                temp_model_path = f"temp_{miner_id}_model.pt"
                torch.save(model.cpu().state_dict(), temp_model_path)

                cashout_url = f"{host_url}/user/cashout"
                params = {"username": miner_id, "seconds_worked": 5.0}

                with open(temp_model_path, "rb") as f:
                    files = {"file": (f"{miner_id}_latest_model.pt", f, "application/octet-stream")}
                    sub_resp = requests.post(cashout_url, params=params, files=files, timeout=10)

                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)

                if sub_resp.status_code == 200:
                    self.points_earned += 10
                    self.root.after(0, lambda: self.points_label.config(text=f"EARNED: {self.points_earned} PTS"))
                    self.log(f"✅ Submission Verified! (Loss: {rounded_loss}) +10 PTS.")
                else:
                    self.log(f"⚠️ Submission rejected with status: {sub_resp.status_code}")

                time.sleep(1)

            except Exception as e:
                self.log(f"❌ Worker Exception: {e}")
                time.sleep(4)

        self.log("⏸️ Worker compute node safely offline.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AthernetWorkerApp(root)
    root.mainloop()
