from pathlib import Path
import os
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
import json

# load environment variable
load_dotenv()


# call openai through groq api key
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

embedding_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"

)

def get_embedding(text):
    response = embedding_client.embeddings.create(model="liquid/lfm-2.5-embedding-350m:free", input=text)

    embedding = response.data[0].embedding

    return embedding


def get_embeddings(texts):
    response = embedding_client.embeddings.create(
        model="liquid/lfm-2.5-embedding-350m:free",
        input=texts
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    return embeddings


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    return dot_product / (magnitude_a * magnitude_b)


# set file path
BASE_DIR = Path(__file__).resolve().parent
knowledge_folder = BASE_DIR / "knowledge"

# empty list to store the content of the txt files
documents = []

# reading the content of the txt files
for file_path in knowledge_folder.glob("*.txt"):
    content = file_path.read_text(encoding="utf-8")
    documents.append({
        "source": file_path.name,
        "content": content
    })

# empty lists for chunks
chunks = []

for document in documents:
    paragraphs = document["content"].split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            chunks.append({
                "source": document["source"],
                "content": paragraph

            })


top_k = 5
similarity_threshold = 0.35


# put chunk contents inside chunk texts list
chunk_texts = [
    f"{chunk['source']}: {chunk['content']}"
    for chunk in chunks
]
# handle a chunk that is an empty string
if not chunk_texts:
    raise ValueError("No knowledge chunks were loaded.")

# get chunk embeddings
def get_chunk_embeddings():
    embeddings_path = BASE_DIR / "chunk_embeddings.json"

    with open(embeddings_path, "r", encoding="utf-8") as file:
        chunk_embeddings = json.load(file)

    if len(chunk_embeddings) != len(chunks):
        raise ValueError(
            "Number of saved embeddings does not match number of knowledge chunks."
        )


    return chunk_embeddings

# a sample list of questions
questions = [
    "Can I come before my scheduled appointment?",

    "I am quite busy now, I will be late for my appointment, what should I do?",

    "We are bringing in an accident victim now, when can we get an appointment with the doctor?",

    "My teeth has been aching me for two days, the pain is now very severe. Can we escalate it to emergency?",

    "Just got a tooth extraction yesterday, can I eat rice and beans now?",

    "My tooth is bleeding, I had an extraction yesterday, what do I do?",

    "How can I get my braces done?",

    "I want to whiten my tooth."
]



# returns relevant context that meets the top results
def build_context(top_results):
    retrieved_chunks = []



    for result in top_results:
        index =  result["index"]
        score = result["score"]
        chunk = chunks[index]


        retrieved_chunks.append({
            "content": chunk["content"],
            "source": chunk["source"],
            "score": score
        })


    context_parts = []



    for retrieved_chunk in retrieved_chunks:
        context_parts.append(
            f"SOURCE: {retrieved_chunk['source']}\n"
            f"CONTENT: {retrieved_chunk['content']}"
        )


    # print(context_parts)

    context = "\n\n".join(context_parts)
    return context

# adds retrieved sources to the sources set
def get_sources(top_results):
    sources = set()

    for result in top_results:
        index = result["index"]
        chunk = chunks[index]

        sources.add(chunk["source"])

    return sources



# send context, and call the llm, then prints question, retrieved context, and llm answer
def call_llm(question, context):
    response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """Answer this dental question
                    
                    Use only the provided context.
                    Do not add information from general knowledge
                    if context doesn't contain enough information,
                    say so
                    
                    Do not diagnose the patient.
                    
                    Every factual claim in the answer must be directly supported
                    by the supplied context.

                    Do not infer policies that are not explicitly stated.

                    Do not assume the opposite of a stated rule.

                    If the context only partially answers the question,
                    answer only the supported part and state that the remaining
                    information is not specified.
                    
                    """
                
                },
                {
                    "role": "user",
                    "content": f"""HOSPITAL CONTEXT:
                    {context}
                    PATIENT QUESTION: {question} """
                    }
            ]
        )


    llm_output = response.choices[0].message.content


    return llm_output
    # print("QUESTION:")
    # print(question)

    # print("\nRETRIEVED CONTEXT:")
    # print(context)

    # print("\nAI ANSWER: ")
    # print(llm_output, "\n")


def retrieve(question, chunk_embeddings):
    question_embedding = get_embedding(question)
    similarities = []
    for index, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(question_embedding, chunk_embedding)
        similarities.append({
            "index": index, "score": score
            })

    similarities_sorted = sorted(similarities, reverse=True, key= lambda item: item["score"])
    top_results = similarities_sorted[:top_k]
    return top_results

def evaluate_thresholds(chunk_embeddings):
    results = []

    for question in questions:
        top_results = retrieve(question, chunk_embeddings)
        best_score = top_results[0]["score"]

        results.append({
            "question": question,
            "score": best_score
        })

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]

    for threshold in thresholds:
        print(f"\nTHRESHOLD: {threshold}")

        for result in results:
            if result["score"] >= threshold:
                status = "PASS"
            else:
                status = "REJECT"

            print(
                f"{result['score']:.4f} | "
                f"{status} | "
                f"{result['question']}"
            )
               
def main():
    chunk_embeddings = get_chunk_embeddings()

    for question in questions:

        top_results = retrieve(
            question,
            chunk_embeddings
        )

        filtered_results = [
            result
            for result in top_results
            if result["score"] >= similarity_threshold
        ]

        if not filtered_results:
            print(f"QUESTION: {question}\n")
            print("RETRIEVAL STATUS:")
            print("No sufficiently relevant context found\n")

        else:
            context = build_context(filtered_results)
            answer = call_llm(question, context)

            print(f"QUESTION: {question}\n")
            print("ANSWER:")
            print(answer)
            print()

if __name__ == "__main__":
    main()