from backend.brain.brain_model import Brain
from llama_cpp import Llama

class QWENBRAIN(Brain):
    # Change default to point to your local downloaded GGUF file path
    def __init__(self, model_name="backend/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"):
        super().__init__(model_name)
        self.model_name = model_name
        self.model = None  # This will hold the Llama instance

        self.system_instruction = (
            "You are the brain behind a robot. Given an input command, you must output a "
            "JSON array containing step-by-step action objects. Do not include any "
            "conversational text, explanations, or markdown code blocks.\n"
            "CRITICAL RULE: If the input command is gibberish, background noise, unrelated "
            "chatter, or does not contain an active request for the robot, you MUST ignore "
            "it entirely and output a completely empty JSON array: []."
        )

    def _load_model(self):
        if self.model is None:
            # Initialize Llama directly with CPU optimization parameters
            self.model = Llama(
                model_path=self.model_name,
                n_ctx=512,       # Limit context window to save RAM
                n_threads=4,     # Restrict to 4 threads so it leaves room for YOLO
                verbose=False    # Keeps terminal clean from heavy debugging logs
            )
        return self.model

    def process_task(self, task_data: str) -> str:
        try:
            model = self._load_model()
        except Exception as exc:
            print(f"Brain model initialization error: {exc}")
            return ""

        # Construct ChatML format directly for Qwen models using string templating
        prompt = (
            f"<|im_start|>system\n{self.system_instruction}<|im_end|>\n"
            f"<|im_start|>user\nTask Input: look around for my car keys<|im_end|>\n"
            f"<|im_start|>assistant\n" + '[{"action": "detect_object", "target": "keys"}]' + "<|im_end|>\n"
            f"<|im_start|>user\nTask Input: Pick up the red ball and place it on the table<|im_end|>\n"
            f"<|im_start|>assistant\n" + '[{"action": "detect_object", "target": "red ball"}, {"action": "pick_up", "target": "red ball"}, {"action": "place_on", "target": "table"}]' + "<|im_end|>\n"
            f"<|im_start|>user\nTask Input: {task_data}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Generate text using highly optimized CPU integer math
        output = model(
            prompt,
            max_tokens=150,
            temperature=0.1,
            stop=["<|im_end|>", "<|im_start|>"] # Stop generating immediately if it hits chat tokens
        )

        response = output["choices"][0]["text"]
        return response.strip()