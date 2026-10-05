from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# load environment variable
load_dotenv()


# call openai through groq api key
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# load the model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")


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
similarity_threshold = 0.45


# put chunk contents inside chunk texts list
chunk_texts = [chunk["content"] for chunk in chunks]

# handle a chunk that is an empty string
if not chunk_texts:
    raise ValueError("No knowledge chunks were loaded.")


# create embeddings from chunk texts
chunk_embeddings = model.encode(chunk_texts)

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



    for score, index in zip(top_results.values, top_results.indices):
        chunk = chunks[index.item()]


        retrieved_chunks.append({
            "content": chunk["content"],
            "source": chunk["source"],
            "score": score.item()
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

    for index in top_results.indices:
        chunk = chunks[index.item()]

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


def retrieve(question):
    question_embedding = model.encode(question)
    
    similarities = cos_sim(question_embedding, chunk_embeddings)[0]
    top_results = similarities.topk(top_k)
    return top_results


def main():
    for question in questions:
        
        top_results = retrieve(question)
        best_score = top_results.values[0].item()
        if best_score < similarity_threshold:
            print(f"QUESTION: {question}\n")
            print("RETRIEVAL STATUS:")
            print("No sufficiently relevant context found\n")
        else:
            context = build_context(top_results)
            call_llm(question, context)

if __name__ == "__main__":
    main()