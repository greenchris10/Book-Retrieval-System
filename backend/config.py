from dotenv import load_dotenv
import os, sys

load_dotenv()

ZILLIZ_CLOUD_URI = os.getenv('ZILLIZ_CLOUD_URI')
ZILLIZ_CLOUD_API_KEY = os.getenv('ZILLIZ_CLOUD_API_KEY')

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BOOKS_JSON_DIR = os.getenv('BOOKS_JSON_DIR')

BERT_MODEL_DIR = os.getenv('BERT_MODEL_DIR')

