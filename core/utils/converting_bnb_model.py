from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

# 1. 모델 경로 설정 (원본 Gemma 모델)
model_path = "LM_Models/gemma-3-12b-it_250724_test"

# 4. 저장할 경로 설정
save_path = "LM_Models/gemma-3-12b-it-bnb4"


# 2. bnb 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # 또는 torch.float16
)

# 3. 모델 로딩 (bnb 적용)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# 4. 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(model_path)


# ✅ 공유 weight 분리 (정확한 경로!)
with torch.no_grad():
    embed = model.language_model.embed_tokens
    head = model.lm_head
    if embed.weight.data_ptr() == head.weight.data_ptr():
        print("→ Shared weight detected. Cloning lm_head.weight to break tying.")
        head.weight = torch.nn.Parameter(head.weight.clone())

# 저장
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
