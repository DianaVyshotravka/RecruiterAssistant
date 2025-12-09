# Resume-Job Matching with Transformers

This repository contains a project that matches resumes to job descriptions using transformer-based models and cosine similarity.

## 📥 Setup Instructions

1. **Download this repository:**

   ```bash
   git clone https://github.com/DianaVyshotravka/RecruiterAssistant.git
   cd RecruiterAssistant
2. **Download the pre-trained model:**

Download the mse_base.pth model file from the following Kaggle notebook: https://www.kaggle.com/code/diankav/sentiment-transformers

Place the mse_base.pth file in the root directory of this repository.

3. **Download pre-embedded dataset:**


4. **Setup PostgreSQL and restore backup**

```bash
docker compose up -d
```
5. **Setup python enviroment**
\
Python 3.11 is required 
\
for Linux
```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
for Windows
```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

6. **Run the app**
\
for Linux
```bash
source .venv/bin/activate && python inference.py
```
for Windows
```bash
.venv\Scripts\activate && python inference.py
```