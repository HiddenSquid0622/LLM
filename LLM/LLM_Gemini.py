#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import json
import urllib.request
import urllib.error
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
import docx
from django.conf import settings


# In[ ]:


API_KEY = settings.GEMINI_API_KEY
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
def call_gemini_generation_api(prompt):
    """Makes an API call to the Gemini text generation model."""
    if not GENERATION_API_URL:
        print("Error: Generation API_URL is not configured. Is the API key missing?")
        return None
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(GENERATION_API_URL, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                response_data = json.loads(response.read().decode('utf-8'))
                return response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            else:
                print(f"Error: Generation API call failed with status {response.status}")
                return None
    except Exception as e:
        print(f"An unexpected error occurred during the generation API call: {e}")
        return None


# In[ ]:


def get_embedding(text):
    """Generates a vector embedding for a given text using the embedding model."""
    if not EMBEDDING_API_URL:
        print("Error: Embedding API_URL is not configured. Is the API key missing?")
        return None
    payload = {"model": f"models/{EMBEDDING_MODEL_NAME}", "content": {"parts": [{"text": text}]}}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(EMBEDDING_API_URL, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                response_data = json.loads(response.read().decode('utf-8'))
                return response_data.get('embedding', {}).get('value')
            else:
                print(f"Error: Embedding API call failed with status {response.status}")
                return None
    except Exception as e:
        print(f"An unexpected error occurred during the embedding API call: {e}")
        return None


# In[ ]:


def load_documents_from_django_dataset(dataset):
    processed_docs = {}
    for doc_data in dataset:
        file_path = doc_data.get("file_path")
        file_type = doc_data.get("file_type")
        content = ""
        try:
            if file_type == 'pdf':
                content = extract_text(file_path)
                print(f"Successfully loaded and processed PDF: {file_path}")
            elif file_type == 'docx':
                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
                print(f"Successfully loaded and processed DOCX: {file_path}")
            elif file_type in ['txt', 'email']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"Successfully loaded and processed Text/Email: {file_path}")
            else:
                print(f"Warning: Unsupported file type '{file_type}' for {file_path}")
                continue
            processed_docs[os.path.basename(file_path)] = content
        except FileNotFoundError:
            print(f"Error: The file was not found at {file_path}")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
    return processed_docs


# In[ ]:


def parse_query_with_llm(query):
    prompt = f"""
    You are an intelligent assistant. Your task is to parse the following user query and extract key details into a structured JSON format.
    The possible keys are "age", "gender", "procedure", "location", and "policy_duration_months".
    If a detail is not present in the query, omit the key from the JSON.
    Your output must be only the JSON object, with no other text or markdown formatting.

    User Query: "{query}"

    JSON Output:
    """
    print("\nParsing Query with Gemini LLM")
    parsed_json_string = call_gemini_api(prompt)

    if not parsed_json_string:
        print("Error: Received no response from LLM for query parsing.")
        return {}

    try:
        cleaned_string = parsed_json_string.strip().replace('```json', '').replace('```', '').strip()
        structured_query = json.loads(cleaned_string)
        print(f"Successfully parsed query into: {structured_query}")
        return structured_query
    except json.JSONDecodeError:
        print(f"Error: LLM did not return valid JSON. Response was: {parsed_json_string}")
        return {}


# In[ ]:


def semantic_search_with_embeddings(query, documents, top_k=3):
    print("\n--- Performing Semantic Search with Vector Embeddings ---")
    chunks = []
    for doc_name, content in documents.items():
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 50]
        for p in paragraphs:
            chunks.append({"source": doc_name, "text": p})
    if not chunks:
        print("Could not create any text chunks from the documents.")
        return []
    print(f"Created {len(chunks)} text chunks from documents.")

    print("Generating embedding for the query...")
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("Failed to generate embedding for the query.")
        return []
    print("Generating embeddings for all document chunks...")
    chunk_embeddings = [get_embedding(chunk['text']) for chunk in chunks]
    valid_embeddings_data = [
        (emb, chunks[i]) for i, emb in enumerate(chunk_embeddings) if emb is not None
    ]
    if not valid_embeddings_data:
        print("Failed to generate any embeddings for the document chunks.")
        return []
    valid_embeddings, valid_chunks = zip(*valid_embeddings_data)
    print("Calculating similarities...")
    query_vec = np.array(query_embedding).reshape(1, -1)
    chunk_vecs = np.array(valid_embeddings)
    similarities = cosine_similarity(query_vec, chunk_vecs)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    relevant_clauses = []
    for index in top_k_indices:
        clause = {
            "source": valid_chunks[index]["source"],
            "clause": valid_chunks[index]["text"],
            "similarity": similarities[index]
        }
        relevant_clauses.append(clause)
        print(f"  - Found relevant clause from '{clause['source']}' (Similarity: {clause['similarity']:.4f}): \"{clause['clause'][:100]}...\"")
        
    return relevant_clauses


# In[ ]:


def evaluate_and_decide_with_llm(structured_query, relevant_clauses):
    query_str = json.dumps(structured_query, indent=2)
    clauses_for_prompt = [{"source": c["source"], "clause": c["clause"]} for c in relevant_clauses]
    clauses_str = "\n\n".join([f"Source: {c['source']}\nClause: {c['clause']}" for c in clauses_for_prompt])
    prompt = f"""
    You are an expert insurance claims processor. Based on the user's details and the provided policy clauses, make a decision.
    Your response MUST be a single, raw JSON object with three keys: "decision" (string: "Approved", "Rejected", or "Further Information Required"),
    "amount" (integer, the approved amount if applicable, otherwise 0), and "justification" (string, explaining the reason and referencing the source and clause).
    Do not include any other text or markdown formatting.

    User Details:
    {query_str}

    Relevant Policy Clauses:
    {clauses_str}

    Decision JSON:
    """
    print("\n--- Evaluating Clauses with Gemini LLM for Final Decision ---")
    decision_json_string = call_gemini_generation_api(prompt)
    if not decision_json_string:
        return {"decision": "Error", "amount": 0, "justification": "Received no response from LLM."}
    try:
        cleaned_string = decision_json_string.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_string)
    except json.JSONDecodeError:
        print(f"Error: LLM did not return valid JSON for the decision. Response was: {decision_json_string}")
        return {"decision": "Error", "amount": 0, "justification": "Failed to parse decision from LLM."}


# In[ ]:


def process_query_pipeline(query, django_dataset):
    if not API_KEY:
        return {"error": "Gemini API key is not configured. Please check your config.json file."}
    documents = load_documents_from_django_dataset(django_dataset)
    if not documents: return {"error": "No documents could be loaded or processed."}
    structured_query = parse_query_with_llm(query)
    if not structured_query: return {"error": "Failed to parse the user query."}
    relevant_clauses = semantic_search_with_embeddings(query, documents)
    if not relevant_clauses:
        return {"decision": "Cannot Determine", "amount": 0, "justification": "No relevant clauses found for the query."}
    final_response = evaluate_and_decide_with_llm(structured_query, relevant_clauses)
    return final_response


# In[ ]:




