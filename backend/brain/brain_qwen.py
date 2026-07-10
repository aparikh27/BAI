from backend.brain.brain_model import Brain
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QWENBRAIN(Brain):
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__(model_name)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        self.system_instruction = (
            "You are the brain behind a robot. Given an input command, you must output a "
            "JSON array containing step-by-step action objects. Do not include any "
            "conversational text, explanations, or markdown code blocks.\n"
            "CRITICAL RULE: If the input command is gibberish, background noise, unrelated "
            "chatter, or does not contain an active request for the robot, you MUST ignore "
            "it entirely and output a completely empty JSON array: []."
        )

    def _load_model(self):
        if self.model is None or self.tokenizer is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        return self.model, self.tokenizer

    def process_task(self, task_data: str) -> str:
        try:
            model, tokenizer = self._load_model()
        except Exception as exc:
            print(f"Brain model initialization error: {exc}")
            return ""

        messages = [
            {"role": "system", "content": self.system_instruction},
            
           
            {"role": "user", "content": "Task Input: look around for my car keys"},
            {"role": "assistant", "content": '[{"action": "detect_object", "target": "keys"}]'}, 

            
            {"role": "user", "content": "Task Input: Pick up the red ball and place it on the table"},
            {"role": "assistant", "content": '[{"action": "detect_object", "target": "red ball"}, {"action": "pick_up", "target": "red ball"}, {"action": "place_on", "target": "table"}]'},              
            
            
            {"role": "user", "content": f"Task Input: {task_data}"}
        ]

       
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=150,  
                temperature=0.1      
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()