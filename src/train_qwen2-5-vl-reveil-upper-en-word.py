import os
from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import lightning as L
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from lightning.pytorch.callbacks import Callback, EarlyStopping
from qwen_vl_utils import process_vision_info

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["ROBOFLOW_API_KEY"] = os.getenv("ROBOFLOW_API_KEY", "")

# === Configuration ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_QLORA = True
MODEL_ID = "/swdata/yin/Cui/EM/reveil/models/qwen2.5-7b-instruct-reveil-en-word-upper/4"
IMAGE_DIR = "/swdata/yin/Cui/Re-Veil/create-dataset-new/dataset_en/en_word_upper_20250513/images"
SYSTEM_MESSAGE = "You are a helpful assistant that can identify English words in images. The image will show only the upper half of an English word, with the lower half masked. Identify the word accurately based on the visible portion.  Please answer with a single word, and do not include any other text."
PROMPT = "The image contains the upper half of an English word. The lower half is masked. What is the word in the image? Please answer with a single word, and do not include any other text. The word is:"

config = {
    "max_epochs": 100,
    "batch_size": 4,
    "lr": 2e-4,
    "check_val_every_n_epoch": 1,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 8,
    "result_path": "models/qwen2.5-7b-instruct-reveil-en-word-upper2"
}

# === Dataset Definition ===
def format_data(image_dir, entry):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "image", "image": os.path.join(image_dir, entry['image'])},
                                     {"type": "text", "text": PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": entry['label']}]}
    ]

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, image_dir):
        self.entries = [json.loads(line) for line in open(jsonl_path)]
        self.image_dir = image_dir

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(os.path.join(self.image_dir, entry['image']))
        return image, entry, format_data(self.image_dir, entry)

train_dataset = JSONLDataset("/swdata/yin/Cui/EM/reveil/data/en/en-word/upper/img_dict_train.json", IMAGE_DIR)
valid_dataset = JSONLDataset("/swdata/yin/Cui/EM/reveil/data/en/en-word/upper/img_dict_val.json", IMAGE_DIR)

# === Model Loading ===
lora_config = LoraConfig(
    lora_alpha=16, lora_dropout=0.05, r=8, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj"], task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_type=torch.bfloat16
) if USE_QLORA else None

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=bnb_config, torch_dtype=torch.bfloat16
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

processor = Qwen2_5_VLProcessor.from_pretrained(
    MODEL_ID, min_pixels=28 * 28, max_pixels=1280 * 28 * 28
)

# === Collate Functions ===
def train_collate_fn(batch):
    _, _, examples = zip(*batch)
    texts = [processor.apply_chat_template(e, tokenize=False) for e in examples]
    images = [process_vision_info(e)[0] for e in examples]
    model_inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = torch.full_like(model_inputs["input_ids"], -100)

    for i, e in enumerate(examples):
        target = e[-1]["content"][0]["text"]
        target_ids = processor.tokenizer.encode(target, add_special_tokens=False)
        for j in range(len(model_inputs["input_ids"][i]) - len(target_ids) + 1):
            if model_inputs["input_ids"][i][j:j+len(target_ids)].tolist() == target_ids:
                labels[i, j:j+len(target_ids)] = torch.tensor(target_ids)
                break
    return model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["pixel_values"], model_inputs["image_grid_thw"], labels

# === Lightning Module ===
class Qwen2_5_Trainer(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels).loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels).loss
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss.item()}

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.config["lr"])

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True, collate_fn=train_collate_fn, num_workers=8)

    def val_dataloader(self):
        return DataLoader(valid_dataset, batch_size=self.config["batch_size"], collate_fn=train_collate_fn, num_workers=8)

# === Callbacks ===
class SaveCheckpoint(Callback):
    def __init__(self, result_path): self.result_path = result_path; self.epoch = 0
    def on_train_epoch_end(self, trainer, pl_module):
        path = f"{self.result_path}/{self.epoch}"; os.makedirs(path, exist_ok=True)
        pl_module.processor.save_pretrained(path); pl_module.model.save_pretrained(path)
        print(f"[Checkpoint] Saved to {path}"); self.epoch += 1
    def on_train_end(self, trainer, pl_module):
        path = f"{self.result_path}/latest"; os.makedirs(path, exist_ok=True)
        pl_module.processor.save_pretrained(path); pl_module.model.save_pretrained(path)
        print(f"[Checkpoint] Final model saved to {path}")

early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)

# === Training ===
model_module = Qwen2_5_Trainer(config, processor, model)

trainer = L.Trainer(
    accelerator="gpu", devices=1,
    max_epochs=config["max_epochs"],
    accumulate_grad_batches=config["accumulate_grad_batches"],
    check_val_every_n_epoch=config["check_val_every_n_epoch"],
    gradient_clip_val=config["gradient_clip_val"],
    log_every_n_steps=10,
    callbacks=[SaveCheckpoint(config["result_path"]), early_stopping],
)

trainer.fit(model_module)
