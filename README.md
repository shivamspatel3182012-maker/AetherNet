# AetherNet
A lightweight, asynchronous distributed machine learning system where a master host streams batch data to worker nodes, collects fine-tuned model weights, and runs automated tournament evaluations to reward top performers.
# Distributed AI Network

A lightweight, decentralized machine learning system built with **FastAPI** and **PyTorch**. The network distributes dynamic data batches to client worker nodes over HTTP/HTTPS, collects trained weight updates, and runs automated evaluation tournaments to update the global base model.

---

## ⚡ Features

* **Central Master Node **: FastAPI server managing validation data, issuing streaming work payloads, logging ledger points, and orchestrating model tournaments.
* **Client Worker Node (`worker.py`)**: Automatic GPU hardware detection (NVIDIA CUDA specs & TFLOPS rating), asynchronous-capable training cycles, and automated weight submission.
* **Dynamic Datasets**: Features and labels are streamed directly into worker memory without persisting intermediate data files to client disk.
* **Automated Tournaments**: Scheduled background jobs verify submitted weights against a central validation set, updating the global base model with winning parameters.

---

## 🛠️ System Architecture
                   ┌─────────────────────────┐
                   │    Master Host Node     │
                   │                         │
                   └────────────┬────────────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       │ Stream Batch Data      │ Download Base Model    │ Upload Trained Weights
       ▼                        ▼                        ▼---

## 🚀 Quickstart

### Prerequisites
* Python 3.9+
* PyTorch
* FastAPI & Uvicorn
* Requests & Pandas
2. Launch a Worker Node
Bash
python worker.py
Enter your network URL (e.g., local address or tunnel URL) and your tracking ID when prompted.
📌 API Endpoints
Method	Endpoint	Description
GET	/task/request	Requests a dynamic batch of data for local training
GET	/model/download	Downloads the current baseline model weights
POST	/user/register	Registers or updates a worker node's hardware specs
POST	/user/cashout	Uploads trained model weights and logs points
GET	/points	Returns the current global points ledger
POST	/admin/manual_tournament	Manually triggers a tournament evaluation cycle
       
