import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, BlipProcessor, BlipForConditionalGeneration
import evaluate
import nltk

# Ensure METEOR dependencies are available
nltk.download('wordnet')
nltk.download('omw-1.4')

# ========== SETTINGS ==========
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

image_folder = './dataset/top200_validation'
ground_truth_file = './captions/top200_validation.json'
output_json = 'caption_results_b1.json'
fine_tuned_model_path = 'blip2_10e-5_200'

prompt = ""
max_tokens = 200
batch_size = 8  # You can adjust based on GPU memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================

# ========== LOAD MODEL ==========
processor = Blip2Processor.from_pretrained(fine_tuned_model_path, use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    fine_tuned_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# ====== Load BLIP ======
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# model.to(device)

model.eval()
# ================================

# ========== LOAD DATA ==========
with open(ground_truth_file, 'r') as f:
    gt_data = json.load(f)

gt_captions = {
    os.path.basename(item["image_path"]): item["caption"]
    for item in gt_data
}
# ===============================

# ========== EVALUATION METRICS ==========
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
# ========================================

results = []
predictions = []
references = []

# ========== LOAD ALL IMAGES ==========
all_images = [
    os.path.join(image_folder, img_name)
    for img_name in os.listdir(image_folder)
    if img_name.lower().endswith((".jpg", ".jpeg", ".png"))
]
# ====================================

# ========== BATCH INFERENCE ==========
for i in tqdm(range(0, len(all_images), batch_size)):
    batch_paths = all_images[i:i + batch_size]
    batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
    
    inputs = processor(images=batch_images, text=[prompt]*len(batch_images), return_tensors="pt", padding=True).to(device)

    # Mixed precision inference
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    
    for j, output in enumerate(outputs):
        pred_caption = processor.tokenizer.decode(output, skip_special_tokens=True).strip()
        image_name = os.path.basename(batch_paths[j])
        ref_caption = gt_captions.get(image_name, "").strip()

        results.append({
            "image_name": image_name,
            "image_path": batch_paths[j],
            "generated_caption": pred_caption,
            "ground_truth_caption": ref_caption
        })

        if ref_caption:
            predictions.append(pred_caption)
            references.append(ref_caption)
# ====================================

# ========== SAVE CAPTIONS ==========
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)
# ===================================

# ========== EVALUATION ==========
meteor_score = meteor.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)

bleu_predictions = [p.split() for p in predictions]
bleu_references = [[r.split()] for r in references]
bleu_score = bleu.compute(predictions=bleu_predictions, references=bleu_references)

# ========== PRINT RESULTS ==========
print("\n Evaluation Metrics:")
print(f"BLEU       : {bleu_score['bleu']:.4f}")
print(f"METEOR     : {meteor_score['meteor']:.4f}")
print(f"ROUGE-L F1 : {rouge_score['rougeL']:.4f}")