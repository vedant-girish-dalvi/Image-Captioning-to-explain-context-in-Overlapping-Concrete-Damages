from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import torch

fine_tuned_model_path = './blip2_10e-5_200'  

# ====== BLIP ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # ====== BLIP2 ======
processor = Blip2Processor.from_pretrained(fine_tuned_model_path, use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained(fine_tuned_model_path, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)
# model = BlipForConditionalGeneration.from_pretrained(fine_tuned_model_path).to(device)
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True, do_rescale=False)
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(img):
    img_input = Image.fromarray(img)
    inputs = processor(img_input, return_tensors="pt").to(device)  # <-- Move to the same device
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

interface = gr.Interface(fn=generate_caption, 
                    inputs=gr.Image(label="Image", placeholder="Image to be captioned"), 
                    outputs=gr.Textbox(lines=5, label="Generated Caption"),
                    title= "Damage Caption Generator",
                    description= "Upload images of Concrete Damages to automatically generate captions."
                    )         
# with gr.Blocks() as interface:
#     gr.Markdown("## 🧠 AI Damage Caption Generator")
#     gr.Markdown("Upload images of Concrete Damages to automatically generate captions.")

#     with gr.Row():
#         inputs=gr.Image(label="Image", placeholder="Image to be captioned")
        
interface.launch()   
