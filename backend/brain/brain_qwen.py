from backend.brain.brain_model import Brain
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QWENBRAIN(Brain):
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__(model_name)
        self.model_name = model_name
        
        # Using float32 for CPU safety, change to "auto" if you have a dedicated GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32, 
            device_map="cpu",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Keep base guidelines/examples stationary in __init__
        self.system_instruction = (
            "You are the brain behind a robot. Given an input command, you must output a "
            "JSON array containing step-by-step action objects. Do not include any "
            "conversational text, explanations, or markdown code blocks (like ```json). "
            "If the input is invalid, return an empty JSON array []."
        )

    def process_task(self, task_data: str) -> str:
        # Build the message history dynamically so the incoming task_data is captured!
        messages = [
            {"role": "system", "content": self.system_instruction},
            
            # Example 1: Single Step (Corrected to valid JSON double quotes)
            {"role": "user", "content": "Task Input: look around for my car keys"},
            {"role": "assistant", "content": '[{"action": "detect_object", "target": "keys"}]'}, 

            # Example 2: Multi-Step (Corrected to a valid JSON array of objects)
            {"role": "user", "content": "Task Input: Pick up the red ball and place it on the table"},
            {"role": "assistant", "content": '[{"action": "detect_object", "target": "red ball"}, {"action": "pick_up", "target": "red ball"}, {"action": "place_on", "target": "table"}]'}, 
            
            # The Live Data appended dynamically right now
            {"role": "user", "content": f"Task Input: {task_data}"}
        ]

        # Apply the Qwen chat template to our fresh messages list
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate the response safely without recording gradients
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=150,  # Trimmed down from 512 since robot steps are short
                temperature=0.1      # Keeps it accurate and deterministic
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()