from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch


processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)


def answer_about_image(image_path: str, question: str) -> str:
    try:
        
        image = Image.open(image_path).convert("RGB")
        prompt = question.strip()

       
        inputs = processor(images=image, text=prompt, return_tensors="pt")

        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate answer
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

        answer = processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return answer if answer else "I couldn't understand the image clearly."

    except Exception as e:
        return f"Error processing image: {str(e)}"
