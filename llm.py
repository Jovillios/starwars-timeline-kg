import ollama


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        ollama.pull(self.model_name)

    def generate(self, prompt: str) -> str:
        response = ollama.generate(self.model_name, prompt) 
        return response["response"]
        

    def summarize(self, text: str) -> str:
        prompt = f"Summarize the following text:\n\n{text}"
        response = self.generate(prompt)
        return response


class Embedding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        ollama.pull(self.model_name)

    def embed(self, text: str) -> list:
        response = ollama.embed(self.model_name, text) 
        return response["embeddings"]


if __name__ == "__main__":
    embedding = Embedding("nomic-embed-text")
    llm = LLM("gemma3:4b")
    response = llm.generate("hello")
    print(response)
    response = embedding.embed("hello")
    print(response)

