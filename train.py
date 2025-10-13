import os
from torch.utils.data import DataLoader
from transformers import AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from dataset import ConstructionDamagesCaptionDataset
from torchvision import transforms
from tqdm import tqdm
from utils import save_model, evaluate_model, compute_bleu
import wandb
from peft import LoraConfig, get_peft_model

def collate_fn(batch):
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]

    encoding = processor(
        images=images,
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    return encoding

# ====== Configuration ======
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ["WANDB_SILENT"] = "true"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
num_epochs = 100
learning_rate = 10e-6
patience = 10
fine_tuned_model_path = './blip2_200_10e-6_100e'  

# ====== WanDB Configuration ======
wandb.login()
run = wandb.init(
    project="blip fine-tuning",
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    }
)

# ====== Load BLIP ======
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# model.to(device)

# ====== BLIP2 ======
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)

# ====== Image Transform (Optional) ======
image_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query","value"],
)

peft_model = get_peft_model(model, config)

# ====== Load Dataset ======
train_dataset = ConstructionDamagesCaptionDataset("./captions/captions.json", processor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# validation_dataset = ConstructionDamagesCaptionDataset("./captions/validation_captions_2.json", processor)
# validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# ====== Optimizer ======
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ====== Training Loop ======
def main(model, num_epochs, train_loader, optimizer, patience):
    print(f"Starting training on {device}...\n")
    
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for idx, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=input_ids
            )
            loss = outputs.loss
            loop.set_postfix(loss=loss.item())
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Training Loss: {avg_epoch_loss:.4f}")

        # Log to wandb
        wandb.log({"training_loss": avg_epoch_loss})

        # Save model if improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_without_improvement = 0
            save_model(fine_tuned_model_path, model, processor)
            print(f"Best model saved with loss: {best_loss:.4f}\n")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)\n")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    wandb.finish()

if __name__ == "__main__":

    main(model=peft_model, num_epochs=num_epochs, train_loader=train_loader, optimizer=optimizer, patience=patience)