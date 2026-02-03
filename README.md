# MLOps: Cats vs Dogs Classifier 🐱 vs 🐶

This project implements an end-to-end MLOps pipeline for binary image classification, featuring a trained CNN model, REST API for inference, and a modern frontend interface.

## 👨‍💻 Developer Information

*   **Name:** Maheshwari Aditya Lalchand
*   **ID/Email:** 2024AA05822@WILP.BITS-PILANI.AC.IN

---

## 🔗 Quick Links (Local)

| Service | URL | Description |
| :--- | :--- | :--- |
| **Frontend** | [http://localhost:5500](http://localhost:5500) | Web Interface for uploading images |
| **Backend API** | [http://localhost:8000](http://localhost:8000) | FastAPI Swagger UI |
| **Health Check** | [http://localhost:8000/health](http://localhost:8000/health) | API Status & Model Load Check |
| **MLFlow UI** | [http://localhost:5001](http://localhost:5001) | Experiment Tracking & Metrics |

---

## 🚀 How to Run Locally

### 1. Start the Backend API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
```bash
cd frontend
python -m http.server 5500
```

### 3. Start MLFlow (Experiment Tracking)
```bash
python -m mlflow ui --host 0.0.0.0 --port 5001
```

---

## 🛠️ CI/CD Pipeline

The project includes a comprehensive **GitHub Actions** workflow with 6 stages:
1.  **Linting**: Code quality check (`flake8`).
2.  **Unit Tests**: Verifies data loading and API logic (`pytest`).
3.  **Model Training**: Trains the CNN and saves artifacts.
4.  **Build**: Creates a Docker container (`mlops-cats-dogs`).
5.  **Scan**: Security vulnerability scanning (`Trivy`).
6.  **Deploy**: Simulates production deployment.

## 📂 Project Structure
```
.
├── .github/workflows/  # CI/CD Pipeline Definition
├── data/               # Raw and processed data (DVC tracked)
├── frontend/           # HTML/CSS/JS Source
├── models/             # Trained Model Artifacts
├── src/                # Python Source Code (API, Training, DataLoader)
├── tests/              # Unit Tests
├── Dockerfile          # Container Definition
└── requirements.txt    # Dependencies
```
