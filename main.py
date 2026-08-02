import os
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "./hf_cache"))

from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
