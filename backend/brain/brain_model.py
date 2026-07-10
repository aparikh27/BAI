from abc import ABC, abstractmethod

class Brain(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def process_task(self, task_data) -> str:
        """Process the given task and return a clean text description string."""
        pass