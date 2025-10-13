import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Save Fine-tuned Model ======
def save_model(model_dir, model, processor):
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    print(f"Model saved to: {model_dir}")
    
def evaluate_model(model, processor, val_json_path, output_csv="generated_captions.csv"):
    model.eval()
    smoothie = SmoothingFunction().method4

    with open(val_json_path, "r") as f:
        val_data = json.load(f)

    results = []

    for entry in tqdm(val_data):
        image_path = entry["image_path"]
        reference_caption = entry["caption"]

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=128)
            predicted_caption = processor.decode(output_ids[0], skip_special_tokens=True)

        # BLEU score
        bleu = sentence_bleu(
            [reference_caption.split()], predicted_caption.split(), smoothing_function=smoothie
        )

        results.append({
            "image_id": entry["image_id"],
            "reference": reference_caption,
            "predicted": predicted_caption,
            "bleu_score": bleu
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")
    print("Average BLEU Score:", df["bleu_score"].mean())



def compute_bleu(model, processor, dataloader, device):
    model.eval()
    # bleu_scores = []
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Compute loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=input_ids
            )
            loss = outputs.loss
            total_val_loss += loss.item()

            # Generate captions
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Ground truths
            ground_truths = processor.batch_decode(input_ids, skip_special_tokens=True)

            # Compute BLEU for each pair
            for reference, hypothesis in zip(ground_truths, generated_texts):
                reference_tokens = word_tokenize(reference.lower())
                hypothesis_tokens = word_tokenize(hypothesis.lower())
    #             bleu = sentence_bleu(
    #                 [reference_tokens],
    #                 hypothesis_tokens,
    #                 smoothing_function=SmoothingFunction().method1
    #             )
    #             bleu_scores.append(bleu)

    # avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss


