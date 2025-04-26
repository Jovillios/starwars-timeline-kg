from llm import Embedding, LLM
import torch
import pandas as pd
from tqdm import tqdm


class EmbeddingBenchmark:
    def __init__(self, embed_model, questions, topk):
        self.embed_model = embed_model 
        self.embeddings = torch.tensor(self.embed_model.load_embeddings())
        self.questions_embeddings = torch.tensor(self.embed_model.embed(questions, from_save=False))
        self.topk = topk
        self.topk_documents_by_questions = self.compute_topk()
        
    def compute_topk(self):
        # Normalize the embeddings for cosine similarity
        doc_norm = self.embeddings / self.embeddings.norm(dim=1, keepdim=True)
        ques_norm = self.questions_embeddings / self.questions_embeddings.norm(dim=1, keepdim=True)

        # Compute cosine similarities
        similarities = ques_norm @ doc_norm.T  # (n_questions, n_documents)

        # Get the top-k documents for each question
        topk_values, topk_indices = torch.topk(similarities, self.topk, dim=1)

        return topk_indices
    
    def __call__(self, event_id):
        # compute score based on the rank of event_id for each questions 
        # product(1 - r_i / top-k) where r_i is the rank of the event_id in the question i 
        # ignore in the product questions where the event_id is not implied 
        rank_by_doc = torch.where(self.topk_documents_by_questions == event_id)[1]
        if len(rank_by_doc) == 0:
            return torch.tensor(0)
        score = torch.mean(1 - rank_by_doc/self.topk) 
        return score


class Benchmark:
    def __init__(self, qa_csv_file, llm, embed_model, topk):
        self.prompt_format = "Answer the following multiple-choice question with only the letter (A, B, C, or D) corresponding to the correct answer. Do not explain your choice. Output only a single letter."
        self.questions, self.answers = self.parse_qa(qa_csv_file) 
        self.llm = llm
        self.embed_model = embed_model
        self.topk = topk

    def parse_qa(self, qa_csv_file):
        df = pd.read_csv(qa_csv_file)
        answers = list(df["Answer"])
        questions = list(df["Question"] + "\nA) " + df["A"] + "\nB) " + df["B"] + "\nC) " + df["C"] + "\nD) " + df["D"])
        return questions, answers

    def get_questions(self):
        return self.questions

    def get_answers(self):
        return self.answers


    def compute_topk(self, doc_embeddings, questions_embeddings):
        # Normalize the embeddings for cosine similarity
        doc_norm = doc_embeddings / doc_embeddings.norm(dim=1, keepdim=True)
        ques_norm = questions_embeddings / questions_embeddings.norm(dim=1, keepdim=True)

        # Compute cosine similarities
        similarities = ques_norm @ doc_norm.T  # (n_questions, n_documents)

        # Get the top-k documents for each question
        topk_values, topk_indices = torch.topk(similarities, self.topk, dim=1)

        return topk_indices[0] # here only 1 question

    def get_context(self, question, descriptions, doc_embeddings):
        question_embeddings = self.embed_model.embed(question, from_save=False)
        idx = self.compute_topk(doc_embeddings, torch.tensor(question_embeddings))
        return [descriptions[i] for i in idx] 


    def get_formatted_answer(self, prompt):
        def format(answer):
            return answer.replace("\n","").replace(".","").strip()
        answer = format(self.llm.generate(prompt))
        i = 0
        MAX = 10
        while not answer in ["A", "B", "C", "D", "NONE"]:
            answer = format(self.llm.generate(prompt))
            print(i)
            i += 1
            if i >= MAX:
                print(f"After {i} attemps, not able to have a well-formatted answer. Last answer: {answer}")
                break
        return answer

    def eval(self, given_answers):
        n = len(self.answers)
        score = 0
        for i in range(n):
            score += (self.answers[i] == given_answers[i])
        return score / n
        

    def __call__(self, descriptions):
        answers = []
        doc_embeddings = torch.tensor(self.embed_model.embed(descriptions, from_save=False)) 
        print("benchmark...")
        for question in tqdm(self.questions):
            question_prompt = f"{self.prompt_format}\n\n{question}"
            context = self.get_context(question, descriptions, doc_embeddings)
            prompt = f"""DOCUMENT: 
{context}

QUESTION: 
{question_prompt}

INSTRUCTIONS:
Answer the user's QUESTION using the DOCUMENT text above. Keep your answer grounded in the facts of the DOCUMENT. If the DOCUMENT doesn’t contain the facts to answer the QUESTION, return NONE.
            """
            answer = self.get_formatted_answer(prompt)
            answers.append(answer)

        # eval 
        score = self.eval(answers)
        return score, answers

