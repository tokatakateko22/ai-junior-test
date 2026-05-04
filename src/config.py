from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "knowledge"
VECTORSTORE_DIR = ROOT_DIR / "data" / "vectorstore"

OSS_MODEL_NAME = os.getenv("OSS_MODEL_NAME", "sshleifer/tiny-gpt2")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 4
