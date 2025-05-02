#### Import Libraries
import fasttext
import numpy as np
import faiss
from sklearn.preprocessing import normalize
import os
import pandas as pd
import re
from random import sample
from sklearn.decomposition import PCA
from random import sample

##### List of 1M e-commerce product names
product_list = pd.read_pickle('/products.pkl')
print(f"Total num of products: {len(product_list)}")
product_list[:3]

# Preparing the training set
def clean_text(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    s = s.strip()
    return s

standardized_names = [clean_text(name) for name in product_list]

output_file = "product_names.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for name in standardized_names:
        f.write(name + "\n")

with open("product_names.txt", "r") as f:
    names = f.read().splitlines()
with open("supervised_training_file.txt", "w") as f:
    for name in names:
        f.write(name + " " + "__label__" + name + "\n")    

#### Training the model in supervised manner
model = fasttext.train_supervised(
    input="supervised_training_file.txt",
    dim=300,
    lr=0.001,
    epoch=100,
    minCount=1,
    minn=3,
    maxn=6,
    neg=5,
    wordNgrams=1,
    loss='ns'
)
model.quantize(qnorm=True)

#### Save the model
model.save_model("fasttext_product_model.bin")
model = fasttext.load_model("fasttext_product_model.bin")

##### Creating embeddings
from multiprocessing import Pool

def get_embedding(name):
    return model.get_word_vector(name)

with Pool(processes=8) as pool:
    embeddings = np.array(pool.map(get_embedding, standardized_names))

# Normalize embeddings for cosine similarity
embeddings = normalize(embeddings)

print(embeddings.shape)

##### Reducing the dimensions of embeddings from 300 to 100
pca = PCA(n_components=100)
embeddings_reduced = pca.fit_transform(embeddings)

# Build FAISS index for fast nearest neighbor search
index = faiss.IndexFlatIP(100)  
index.add(embeddings_reduced)  

##### Saving the index
faiss.write_index(index, "faiss_index.index")

##### Getting the matching results from FAISS
def correct_spelling_faiss(input_name, model, index, top_k=5):
    input_clean = clean_text(input_name)
    input_vector = model.get_word_vector(input_clean)
    input_vector = normalize(input_vector.reshape(1, -1))
    input_vector = pca.transform(input_vector)
    distances, indices = index.search(input_vector, top_k)
    return [(product_list[i], standardized_names[i], distances[0][j]) for j, i in enumerate(indices[0])]

##### Filtering the results from FAISS using Levenshtein Distance
def correct_spelling_fuzzy_logic_LV_distance(misspelled_input, candidates_with_faiss, top_k=5):
    from Levenshtein import distance
    input_clean = clean_text(misspelled_input)
    candidates_with_fuzzy = [
        (name, std_name, faiss_score, distance(input_clean, std_name))
        for name, std_name, faiss_score in candidates_with_faiss
    ]
    candidates_with_fuzzy.sort(key=lambda x: (x[3], x[2]))
    return candidates_with_fuzzy[:top_k]

##### Filtering the results from FAISS using Jaro-Winkler Similarity
def correct_spelling_fuzzy_logic_jw_sim(misspelled_input, candidates_with_faiss, fuzzy_threshold=0.1, top_k=5):
    from jellyfish import jaro_winkler_similarity
    input_clean = clean_text(misspelled_input)
    candidates_with_fuzzy = [
        (name, std_name, faiss_score, jaro_winkler_similarity(input_clean, std_name))
        for name, std_name, faiss_score in candidates_with_faiss
    ]
    filtered_candidates = [(n, f, s, jw) for n, f, s, jw in candidates_with_fuzzy if jw >= fuzzy_threshold]
    filtered_candidates.sort(key=lambda x: (x[3], x[2]), reverse=True)
    return filtered_candidates[:top_k]

##### Inference pipeline
def correct_spelling(query, top_k):
    candidates_with_faiss = correct_spelling_faiss(query, model, index, top_k=200)
    lv_predicted_names = correct_spelling_fuzzy_logic_LV_distance(query, candidates_with_faiss, top_k=top_k)
    jw_predicted_names = correct_spelling_fuzzy_logic_jw_sim(query, candidates_with_faiss, fuzzy_threshold=0.1, top_k=top_k)
    
    lv_predicted_names = [item[0] for item in lv_predicted_names]
    jw_predicted_names = [item[0] for item in jw_predicted_names]
    
    return lv_predicted_names, jw_predicted_names

qs = ['aple', 'GGOOlE', 'applr', 'googlr']
for q in qs:
    print(correct_spelling(q, 5))

##### Evaluation
#### Generate Misspellings
test_queries = ['Apple Watch', 'Google Pixel', 'Microsoft Surface']
def accuracy_at_k(model, index, test_queries, top_k, k_values=[1, 5, 10]):
    lv_results = {k: 0 for k in k_values}
    jw_results = {k: 0 for k in k_values}
    ngram_results = {k: 0 for k in k_values}
    
    for query, target in test_queries:
        lv_predicted_names, jw_predicted_names = correct_spelling(query, top_k=top_k)
        for k in k_values:
            if target in lv_predicted_names[:k]:
                lv_results[k] += 1
            if target in jw_predicted_names[:k]:
                jw_results[k] += 1
    
    n_queries = len(test_queries)
    lv_accuracy = {k: lv_results[k] / n_queries for k in k_values}
    jw_accuracy = {k: jw_results[k] / n_queries for k in k_values}
    ngram_accuracy = {k: ngram_results[k] / n_queries for k in k_values}
    return lv_accuracy, jw_accuracy, ngram_accuracy

lv_accuracy, jw_accuracy = accuracy_at_k(model, index, test_queries, top_k=10, k_values=[1, 5, 10])

print("LV Accuracy@K:")
for k, acc in lv_accuracy.items():
    print(f"LV Accuracy@{k}: {acc:.4f}")

print("-"*50)
print("JW Accuracy@K:")
for k, acc in jw_accuracy.items():
    print(f"JW Accuracy@{k}: {acc:.4f}")

print("-"*50)

#### MRR: Higher the better
def mrr(model, index, test_queries, k):
    lv_reciprocal_ranks = []
    jw_reciprocal_ranks = []
    ngram_reciprocal_ranks = []
    
    for query, target in test_queries:
        lv_predicted_names, jw_predicted_names = correct_spelling(query, top_k=k)
        try:
            rank = lv_predicted_names.index(target) + 1
            lv_reciprocal_ranks.append(1 / rank)
        except ValueError:
            lv_reciprocal_ranks.append(0)
        try:
            rank = jw_predicted_names.index(target) + 1
            jw_reciprocal_ranks.append(1 / rank)
        except ValueError:
            jw_reciprocal_ranks.append(0)
    
    lv_mrr_score = np.mean(lv_reciprocal_ranks) if lv_reciprocal_ranks else 0
    jw_mrr_score = np.mean(jw_reciprocal_ranks) if jw_reciprocal_ranks else 0
    return lv_mrr_score, jw_mrr_score

lv_mrr_score, jw_mrr_score = mrr(model, index, test_queries, k=1)

print(f"lv_mrr_score: {lv_mrr_score:.4f}")
print(f"jw_mrr_score: {jw_mrr_score:.4f}")
