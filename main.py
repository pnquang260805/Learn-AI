import os

from dotenv import load_dotenv

from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
