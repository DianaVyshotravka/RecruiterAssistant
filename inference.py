import os
import re
import ast
import logging
import argparse
import warnings

import pandas as pd
import numpy as np
import requests
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from bs4 import BeautifulSoup
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "princeton-nlp/sup-simcse-roberta-base"
BATCH_SIZE = 32
EPOCHS = 15
MAX_LENGTH = 256
MODEL_PATH = "mse_base.pth"

# Database Configuration
DB_NAME = os.getenv("DB_NAME", "vectordb")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")


class SiameseSimCSE(nn.Module):
    def __init__(self, model_name, freeze_percentage=0.8):
        super(SiameseSimCSE, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        total_layers = len(self.encoder.encoder.layer)
        layers_to_freeze = int(total_layers * freeze_percentage)

        for param in self.encoder.parameters():
            param.requires_grad = False

        for i in range(layers_to_freeze, total_layers):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class VectorDB:
    def __init__(
        self,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    ):
        try:
            self.connection = psycopg2.connect(
                dbname=dbname, user=user, password=password, host=host, port=port
            )
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            # We don't raise here to allow the app to start even if DB is down,
            # but methods will fail. Alternatively, we could raise.
            self.connection = None

    def load_candidates(self, df):
        if not self.connection:
            return "Database not connected"
        cur = self.connection.cursor()
        try:
            for _, row in df.iterrows():
                cur.execute(
                    "INSERT INTO candidates (resume_text, embedding_vector) VALUES (%s, %s)",
                    (row["resume_text"], row["resume_embeddings"]),
                )
            self.connection.commit()
            return f"{len(df)} records saved to candidates table."
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error loading candidates: {e}")
            return f"Error: {e}"
        finally:
            cur.close()

    def save_feedback(self, df):
        if not self.connection:
            return "Database not connected"
        cur = self.connection.cursor()
        try:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO feedback (job_description_text, resume_text, label)
                    VALUES (%s, %s, %s)
                    """,
                    (row["vacancy"], row["candidate"], row["label"]),
                )
            self.connection.commit()
            return f"{len(df)} records saved to feedback table."
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error saving feedback: {e}")
            return f"Error: {e}"
        finally:
            cur.close()

    def insert_vacancy(self, vacancy_text, vacancy_embedding, vacancy_link=None):
        if not self.connection:
            raise Exception("Database not connected")
        cur = self.connection.cursor()
        try:
            cur.execute(
                """
                INSERT INTO vacancies (vacancy_text, vacancy_link, embedding_vector)
                VALUES (%s, %s, %s)
                RETURNING vacancy_id
                """,
                (vacancy_text, vacancy_link, vacancy_embedding),
            )
            vacancy_id = cur.fetchone()[0]
            self.connection.commit()
            return vacancy_id
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error inserting vacancy: {e}")
            raise
        finally:
            cur.close()

    def find_similar_candidates(self, k, vacancy_id):
        if not self.connection:
            return []
        cur = self.connection.cursor()
        try:
            cur.execute(
                "SELECT embedding_vector FROM vacancies WHERE vacancy_id = %s",
                (vacancy_id,),
            )
            result = cur.fetchone()
            if not result:
                return "Vacancy not found"

            vacancy_embedding = result[0]
            cur.execute(
                """
                SELECT (1 - (embedding_vector <=> %s)) AS similarity, resume_text
                FROM candidates
                ORDER BY similarity DESC
                LIMIT %s
                """,
                (vacancy_embedding, k),
            )
            results = cur.fetchall()
            return results
        except Exception as e:
            logger.error(f"Error finding candidates: {e}")
            return []
        finally:
            cur.close()

    def close(self):
        if self.connection:
            self.connection.close()


def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"https?://\S+|www\.\S+", "<url>", text)
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002500-\U00002bef"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u200d"
        "\u2640-\u2642"
        "\u2600-\u2b55"
        "\u23cf\u23e9\u231a"
        "\u3030"
        "\ufe0f\u2069\u2066"
        "\u200c\u2068\u2067"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    text = re.sub(r"[\xa0\u200d\t\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"[^\w\s\.,'!?]", "", text)

    return text


def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Error during loading web-page: {str(e)}"

    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(
        ["script", "style", "header", "footer", "nav", "aside", "form", "noscript"]
    ):
        tag.decompose()
    main_content = soup.find("main")
    if not main_content:
        main_content = max(
            soup.find_all("div"),
            key=lambda d: len(d.get_text(strip=True)),
            default=None,
        )

    if main_content:
        text = main_content.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)

    return text


def extract_text(input_type, text_input=None, url=None, pdf_file=None):
    if input_type == "from url":
        try:
            text = extract_text_from_url(url)
        except Exception as e:
            return f"Error fetching URL: {e}"
    elif input_type == "from text":
        text = text_input
    else:
        return "Invalid input type"

    return text


class InferenceEngine:
    def __init__(self, model_name=MODEL_NAME, model_path=MODEL_PATH, device=DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SiameseSimCSE(model_name).to(device)

        if os.path.exists(model_path):
            logger.info(f"Loading model weights from {model_path}")
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
        else:
            logger.warning(
                f"Model weights file {model_path} not found. Using base model weights."
            )

        self.model.to(device)
        self.model.eval()

    def get_embedding(self, input_type, text_input=None, url=None, pdf_file=None):
        db = VectorDB()

        raw_text = extract_text(input_type, text_input, url, pdf_file)

        if not isinstance(raw_text, str):
            return raw_text, None

        cleaned_text = preprocess_text(raw_text)

        encoded = self.tokenizer(
            cleaned_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            embedding = self.model(input_ids, attention_mask)

        url_val = url if url else ""

        embedding_list = embedding[0].cpu().numpy().tolist()
        try:
            vacancy_id = db.insert_vacancy(cleaned_text, embedding_list, url_val)
        except Exception as e:
            return f"Database Error: {e}", None
        finally:
            db.close()

        return cleaned_text, vacancy_id


def create_gradio_interface(engine):

    def interface_submit_vacancy(input_type, text_input, url_input):
        cleaned_text, vacancy_id = engine.get_embedding(
            input_type=input_type,
            text_input=text_input if input_type == "from text" else None,
            url=url_input if input_type == "from url" else None,
        )
        if vacancy_id is None:
            return cleaned_text, "Error", "Failed to encode"
        return cleaned_text, vacancy_id, "Data has been successfully encoded"

    def interface_find_candidates(k, vacancy_id):
        k = int(k)
        db = VectorDB()
        candidates = db.find_similar_candidates(k, vacancy_id)
        db.close()

        if isinstance(candidates, str):  # Error message
            return []

        results = [(round(sim, 4), text) for sim, text in candidates]
        return results

    def save_labels(cleaned_text, *values):
        db = VectorDB()
        labels = values[:10]
        candidates = values[10:]

        data = []
        for i in range(10):
            if candidates[i] and candidates[i].strip():
                data.append(
                    {
                        "vacancy": cleaned_text,
                        "candidate": candidates[i],
                        "label": labels[i],
                    }
                )

        df = pd.DataFrame(data)
        result = db.save_feedback(df)
        db.close()
        return result

    with gr.Blocks() as demo:
        gr.Markdown("# Recruiter Assistant")

        input_type = gr.Radio(
            choices=["from url", "from text"], label="Input type", value="from text"
        )

        with gr.Row():
            text_input = gr.Textbox(
                label="Type/insert vacancy text here", lines=8, visible=True
            )
            url_input = gr.Textbox(label="Insert vacancy URL here", visible=False)

        generate_btn = gr.Button("Get embedding")
        cleaned_text_output = gr.Textbox(label="Cleaned text", lines=8, visible=False)
        # embedding_output = gr.Textbox(visible=False) # Unused in original
        status_output = gr.Textbox(
            visible=True, interactive=False, label="Embedding status"
        )
        vacancy_id_output = gr.Textbox(label="Vacancy ID")

        def toggle_inputs(choice):
            return (
                gr.update(visible=(choice == "from text")),
                gr.update(visible=(choice == "from url")),
                gr.update(visible=(choice == "from url")),
            )

        input_type.change(
            toggle_inputs,
            inputs=[input_type],
            outputs=[text_input, url_input, cleaned_text_output],
        )

        generate_btn.click(
            interface_submit_vacancy,
            inputs=[input_type, text_input, url_input],
            outputs=[cleaned_text_output, vacancy_id_output, status_output],
        )

        gr.Markdown("## Find Candidates")
        k_input = gr.Number(label="Number of top-k candidates", value=5, precision=0)
        find_btn = gr.Button("Find Candidates")

        candidate_rows = []
        similarity_bars = []
        candidate_boxes = []
        label_radios = []

        for i in range(10):
            with gr.Column(visible=False) as column:
                similarity = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.0001,
                    label=f"Similarity {i+1}",
                    interactive=False,
                )
                text = gr.Textbox(label=f"Candidate {i+1}", lines=5, interactive=False)
                radio = gr.Radio(
                    choices=["Fit", "No Fit"],
                    label="Assessment",
                    value="Fit",
                    interactive=True,
                )
                similarity_bars.append(similarity)
                candidate_boxes.append(text)
                label_radios.append(radio)
                candidate_rows.append(column)

        def update_candidate_outputs(k, vacancy_id_output):
            results = interface_find_candidates(k, vacancy_id_output)
            updates = []

            for i in range(10):
                if i < len(results):
                    sim, text = results[i]
                    updates += [
                        gr.update(visible=True),
                        gr.update(value=sim),
                        gr.update(value=text),
                        gr.update(value="Fit"),
                    ]
                else:
                    updates += [
                        gr.update(visible=False),
                        gr.update(value=0.0),
                        gr.update(value=""),
                        gr.update(value="Fit"),
                    ]
            return updates

        find_btn.click(
            update_candidate_outputs,
            inputs=[k_input, vacancy_id_output],
            outputs=sum(
                [
                    [
                        candidate_rows[i],
                        similarity_bars[i],
                        candidate_boxes[i],
                        label_radios[i],
                    ]
                    for i in range(10)
                ],
                [],
            ),
        )

        gr.Markdown("## Save Feedback")
        save_btn = gr.Button("Save Feedback")
        saved_results_output = gr.Textbox(label="Saved Results Status", lines=1)

        save_btn.click(
            save_labels,
            inputs=[cleaned_text_output] + label_radios + candidate_boxes,
            outputs=saved_results_output,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Recruiter Assistant Inference Service"
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run Gradio on")
    parser.add_argument("--share", action="store_true", help="Share Gradio link")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    logger.info("Starting Inference Engine...")
    engine = InferenceEngine()

    logger.info("Launching Gradio Interface...")
    demo = create_gradio_interface(engine)
    demo.launch(server_port=args.port, share=args.share, debug=args.debug)


if __name__ == "__main__":
    main()
