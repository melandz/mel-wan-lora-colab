# 🐱 MEL WAN LoRA 訓練 - 一鍵 Colab

## 檔案準備完成
所有檔案在 `D:/Users/s8823/Desktop/melandzone/colab_training/`

## 🚀 三步驟

### 1️⃣ 上傳到 Google Drive
把以下兩個檔案上傳到你的 Google Drive 根目錄 (MyDrive)：
- `lora_trainer.tar.gz` (12MB)
- `mel_wan_dataset.tar.gz` (16MB)

### 2️⃣ 打開 Colab
點擊這個連結 → https://colab.research.google.com/
上傳 `MEL_WAN_LoRA_Training.ipynb` 或直接新建貼上代碼

### 3️⃣ 依序執行
每個 cell 按 Shift+Enter：
- Cell 1: 掛載 Drive + 安裝依賴
- Cell 2: 解壓訓練框架 + 資料集  
- Cell 3: 下載 WAN 2.2 模型 (~15分鐘)
- Cell 4: 開始訓練 🔥 (~2小時)
- Cell 5: LoRA 自動存回 Drive

## 📦 產出
訓練完成後在 Google Drive 會看到：
`MyDrive/mel_wan_lora_output/mel_wan_lora_v1.safetensors`

---

## 🔧 Colab 完整代碼（貼到 Colab 第一個 cell）

```python
#@title 🐱 1. 掛載 Drive + 環境安裝
from google.colab import drive
drive.mount('/content/drive')
%cd /content

# GPU 確認
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 安裝依賴
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q accelerate transformers sentencepiece protobuf einops huggingface_hub safetensors peft bitsandbytes

# 解壓訓練框架
!tar xzf /content/drive/MyDrive/lora_trainer.tar.gz -C /content/
!ls /content/lora_trainer/

# 解壓資料集
!tar xzf /content/drive/MyDrive/mel_wan_dataset.tar.gz -C /content/
!ls /content/images/ | wc -l
!ls /content/captions/ | wc -l

print("✅ 環境就緒")
```

```python
#@title 📥 2. 下載 WAN 2.2 模型 (~28GB)
import os
MODEL_DIR = '/content/wan22_model'
QWEN3_DIR = '/content/qwen3_06b'

# 下載 WAN 2.2 DiT
!huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
  --local-dir {MODEL_DIR} \
  --include "*.safetensors" --include "*.json"

# 下載 Qwen3 0.6B
!huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir {QWEN3_DIR}

print("✅ 模型就緒")
!ls -lh {MODEL_DIR}/ | head -5
```

```python
#@title 📝 3. 建立訓練配置
import json, os
from pathlib import Path

# dataset.json
metadata = []
for img in sorted(Path('/content/images').glob('*')):
    if img.suffix.lower() not in ['.png','.jpg','.jpeg']: continue
    # 找對應標註
    cap_name = img.stem.split('_',1)[-1] + '.txt' if '_' in img.stem else img.stem + '.txt'
    cap_files = list(Path('/content/captions').glob('*' + cap_name[-20:]))
    cap_path = cap_files[0] if cap_files else None
    caption = cap_path.read_text().strip() if cap_path else "mel_character"
    metadata.append({"image": str(img), "caption": caption})
with open('/content/dataset.json','w') as f:
    for m in metadata: f.write(json.dumps(m,ensure_ascii=False)+'\n')
print(f"Dataset: {len(metadata)} images")

# train_config.toml
config = f'''
pretrained_model_name_or_path = "{MODEL_DIR}"
qwen3 = "{QWEN3_DIR}"
vae = "{MODEL_DIR}/vae"
output_dir = "/content/drive/MyDrive/mel_wan_lora_output"
output_name = "mel_wan_lora_v1"
dataset_config = "/content/dataset.json"
resolution = "832,480"
batch_size = 1
max_train_epochs = 4
save_every_n_epochs = 1
network_dim = 8
network_alpha = 16
learning_rate = 1e-4
lr_scheduler = "cosine"
lr_warmup_steps = 50
optimizer_type = "adamw8bit"
mixed_precision = "bf16"
gradient_checkpointing = true
blocks_to_swap = 28
cpu_offload_checkpointing = true
cache_latents = true
cache_text_encoder_outputs = true
attn_mode = "torch"
split_attn = false
discrete_flow_shift = 5.0
weighting_scheme = "none"
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.0
'''
with open('/content/train_config.toml','w') as f: f.write(config)
print("✅ 配置就緒")
```

```python
#@title 🔥 4. 開始訓練！
import sys
sys.path.insert(0, '/content/lora_trainer')

!python /content/lora_trainer/anima_train_network.py \
  --config_file /content/train_config.toml \
  2>&1 | tee /content/train_log.txt

print("\n✅ 完成！LoRA 在 MyDrive/mel_wan_lora_output/")
```
