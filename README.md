# AI Doctor Helper (MediKiosk - SIH26047) — Project Overview

## 📌 Purpose of the Project & What It Does

**AI Doctor Helper (MediKiosk)** is an intelligent Clinical Decision Support & Smart Triage System created to assist doctors, triage nurses, and healthcare facilities. It automates patient intake, safety triage, medical document extraction, clinical decision support, and administrative analytics.

### Key Functionalities:
1. **Adaptive Patient Intake & NLU (AI-1)**:
   - Interactive intake interview with Natural Language Understanding (NLU) to capture chief complaints, symptoms, and medical history.
   - Voice interaction support using speech-to-text (Faster-Whisper) and text-to-speech (Edge-TTS).
   - Conversation repair to handle unclear or incomplete patient inputs.

2. **Clinical Safety & Triage Engine (AI-2 & AI-5E)**:
   - Real-time priority assessment (Emergency, High, Medium, Low) and red-flag detection based on safety rules.
   - Dynamic patient routing to relevant departments (Emergency, Cardiology, General Medicine, OPD, etc.).

3. **Medical Document Processing & Explainable AI (AI-3)**:
   - Multi-format document intake (PDF, Images, Lab Reports, Prescriptions, Discharge Summaries).
   - Text extraction (OCR via `pypdf`, `Pillow`, `pytesseract`).
   - Explainable document classification, entity extraction, verification queue, and timeline building.

4. **Clinical Decision Support System / CDSS (AI-4)**:
   - Clinical risk evaluation, medication interaction checks, diagnostic investigation recommendation, consultation copilot, and AI question assistant for clinicians.

5. **Doctor Workspace & Hospital Analytics (AI-5)**:
   - Patient encounter queue management, consultation workspace, clinical handoffs, and real-time hospital analytics dashboard.

---

## 🏗️ Project Structure

```text
AI_Doctor_Helper-/
├── .gitignore                       # Git ignore configuration
├── OverView.md                      # Complete project overview & setup guide
│
├── backend/                         # FastAPI Python Backend
│   ├── main.py                      # Core unified FastAPI application server
│   ├── requirements.txt             # Python backend dependencies
│   ├── sih26047.db                  # SQLite database (auto-created on server launch)
│   │
│   ├── ai1f_orchestrator.py         # AI-1: Patient intake orchestration
│   ├── clinical_nlu.py              # AI-1: Clinical NLU entity & symptom extraction
│   ├── conversation_repair.py       # AI-1: Patient input clarification logic
│   ├── ai2_safety.py                # AI-2: Safety rule engine & red-flag triage
│   ├── ai3a/                        # AI-3A: Document intake module
│   ├── ai3c_document_classifier.py  # AI-3C: Explainable document classification
│   ├── ai3d_medical_extractor.py    # AI-3D: Structured medical data extraction
│   ├── ai3e_document_verification.py # AI-3E: Document verification queue
│   ├── ai3f_clinical_timeline.py    # AI-3F: Clinical timeline aggregator
│   ├── ai3g_explainability.py       # AI-3G: Explainability report generator
│   ├── ai3h_clinical_handoff.py     # AI-3H: Clinical handoff generator
│   ├── ai4a_clinical_summary.py     # AI-4A: Clinical summary engine
│   ├── ai4b_clinical_risk.py        # AI-4B: Clinical risk assessment
│   ├── ai4c_clinical_decision_support.py # AI-4C: Decision support logic
│   ├── ai4d_medication_intelligence.py   # AI-4D: Medication intelligence
│   ├── ai4e_investigation_intelligence.py # AI-4E: Investigation intelligence
│   ├── ai4f_clinical_question_assistant.py# AI-4F: Clinical Q&A assistant
│   ├── ai4g_consultation_copilot.py      # AI-4G: Consultation copilot
│   ├── ai4h_final_clinical_gate.py       # AI-4H: Final clinical safety gate
│   ├── ai5b_encounter_queue.py           # AI-5B: Encounter queue management
│   ├── ai5c_doctor_workspace.py          # AI-5C: Doctor workspace payload builder
│   ├── ai5d_consultation.py              # AI-5D: Consultation note generator
│   ├── ai5e_triage.py                    # AI-5E: Smart triage backend
│   ├── ai5g_analytics.py                 # AI-5G: Analytics dashboard engine
│   ├── phase5a_integration_audit.py      # Integration health audit
│   │
│   ├── uploads/                     # Storage for uploaded documents & voice clips
│   ├── test_*.py                    # Automated test files (Pytest)
│   └── venv/                        # Python virtual environment (git-ignored)
│
└── frontend/                        # React + Vite Frontend
    ├── index.html                   # HTML entry point
    ├── package.json                 # Node dependencies and scripts
    ├── src/
    │   ├── main.jsx                 # React root renderer
    │   ├── App.jsx                  # Main UI & role-based dashboard views
    │   └── styles.css               # Application stylesheet
    └── node_modules/                # Node packages (git-ignored)
```

---

## 🚀 Step-by-Step Guide to Run locally

Follow these instructions to run the application on a local machine after cloning the repository.

### 📋 Prerequisites
Ensure you have the following installed on your machine:
- **Git**
- **Python 3.10+**
- **Node.js (v18+) & npm**
- *(Optional)* **Tesseract OCR** (for image text extraction)

---

### Step 1: Clone the Repository
Open your terminal and run:
```bash
git clone <repository-url>
cd AI_Doctor_Helper-
```

---

### Step 2: Set Up and Start the Backend

1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   - **Windows (PowerShell / CMD)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install required Python packages:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Launch the FastAPI server:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```
   - Backend Server: `http://127.0.0.1:8000`
   - Interactive API Docs (Swagger UI): `http://127.0.0.1:8000/docs`

---

### Step 3: Set Up and Start the Frontend

1. Open a **new terminal tab/window** and navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```

2. Install Node dependencies:
   ```bash
   npm install
   ```

3. Start the dev server:
   ```bash
   npm run dev
   ```
   - Frontend Application: `http://localhost:5173` (or the URL shown in terminal output).

---

### 🧪 Step 4: Run Automated Tests (Optional)

To execute the test suite across all AI modules:

```bash
cd backend
pytest
```
