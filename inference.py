import torch
from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image
import matplotlib.pyplot as plt

class SimpleInference:
    """
    A class for performing inference using Hugging Face models with optional LoRA adapters.
    Supports both local images and image URLs as input.
    """
    
    def __init__(self, model_id="BAAI/RoboBrain", lora_id=None):
        """
        Initialize the model and processor.
        
        Args:
            model_id (str): Path or Hugging Face model identifier (default: "BAAI/RoboBrain")
            lora_id (str, optional): Path or Hugging Face model for LoRA weights. Defaults to None.
        """
        print("Loading Checkpoint ...")
        self.model = AutoModelForPreTraining.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, 
        ).to("cuda:0")

        self.processor = AutoProcessor.from_pretrained(model_id)

        # If LoRA weights are provided, load and adapt the base model
        if lora_id is not None:
            from peft import PeftModel
            print("Loading LoRA Weights...")
            self.processor.image_processor.image_grid_pinpoints = [[384, 384]]
            self.model.base_model.base_model.config.image_grid_pinpoints = [[384, 384]]
            self.model = PeftModel.from_pretrained(self.model, lora_id)
            print(f"Model is initialized with {model_id} and {lora_id}.")
        else:
            print(f"Model is initialized with {model_id}.")
        
    def inference(self, text, image, do_sample=True, temperature=0.7):
        """Perform inference with text and image input."""
        if image.startswith("http"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image", "url": image},
                    ],
                },
            ]
        elif isinstance(image, Image.Image) or isinstance(image, str):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image", "image": Image.open(image) if isinstance(image, str) else image},
                    ],
                },
            ]

        print("Processing input...")
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        print("Generating output...")
        output = self.model.generate(**inputs, max_new_tokens=250, do_sample=do_sample, temperature=temperature)
        
        prediction = self.processor.decode(
            output[0][2:],
            skip_special_tokens=True
        ).split("assistant")[-1].strip()

        return prediction

    def draw_bbox(self, image, bbox, color='red', label=''):
        """Draw bounding box on the image."""
        x1, y1, x2, y2 = bbox
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2, label=label))

    def visualization(self, bbox, image, prompt=None):
        if isinstance(bbox, str):
            x1, y1, x2, y2 = eval(bbox)
        else:
            x1, y1, x2, y2 = bbox

        img = Image.open(image)
        img_w, img_h = img.size
        
        abs_bbox = [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        self.draw_bbox(img, abs_bbox, color='blue', label='Predicted')

        if prompt:
            plt.title(prompt)
        plt.axis('off')
        plt.legend()
        plt.show()

        plt.savefig("demo_visualization.png", bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":
    model_id="/home/vlm/pretrain_model/robobrain_baai_hf"  # "BAAI/RoboBrain"
    lora_id="/home/vlm/workspace/checkpoints/hf_lora_new_exp_1" # "BAAI/RoboBrain-LoRA-Affordance"
    model = SimpleInference(model_id, lora_id)

    # Example 1:
    prompt = "You are a robot using the joint control. The task is \"pick_up the suitcase\". Please predict a possible affordance area of the end effector?"

    image = "./assets/demo/affordance_1.jpg"

    pred = model.inference(prompt, image, do_sample=False)
    print(f"Prediction: {pred}")
    model.visualization(pred, image, prompt)
