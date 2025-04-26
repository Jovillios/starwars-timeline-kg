import ollama
import pickle
import os


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        ollama.pull(self.model_name)

    def generate(self, prompt: str) -> str:
        response = ollama.generate(self.model_name, prompt)
        return response["response"]

    def summarize(self, descriptions: list) -> str:
        prompt = f"Summarize the following descriptions of event nodes into a single event node description of one sentence :\n\n{'\n\n'.join(descriptions)}"
        response = self.generate(prompt)
        return response


class Embedding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        ollama.pull(self.model_name)
        self.file_name = "embeddings.pkl"

    def load_embeddings(self):
        """Load the embeddings previously generated."""
        embeddings = pickle.load(open(self.file_name, "rb"))
        return embeddings

    def embed(self, text: str, from_save=True) -> list:
        """Generate the embeddings for the given text."""
        if from_save and os.path.exists(self.file_name):
            # Load the embeddings from the file
            return self.load_embeddings()
        # Generate the embeddings
        response = ollama.embed(self.model_name, text)
        # Save the embeddings to a file
        if from_save:
            with open(self.file_name, "wb") as f:
                pickle.dump(response["embeddings"], f)
        return response["embeddings"]

