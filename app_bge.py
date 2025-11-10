# coding=utf-8
from flask import Flask,request, jsonify
from FlagEmbedding import FlagLLMModel,BGEM3FlagModel,FlagLLMReranker
from transformers import AutoTokenizer
import torch
import json
import os
import gc
import hashlib
from functools import wraps

os.environ["CUDA_VISIBLE_DEVICES"]="0"
app = Flask(__name__)
# lilly_ai
api_key = "aceab8b0263799ed88cb7ee3eee878b0"

def api_key_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        key = request.json.get('api_key')
        if not key or api_key!=hashlib.md5(key.encode()).hexdigest():
            return jsonify({'error': 'Invalid or missing API key'}), 403
        return f(*args, **kwargs)
    return wrapper

# model_path = '/home/dev/med_search/BAAI/bge-m3'
model_path = 'BAAI/bge-m3'
model = BGEM3FlagModel(model_path,use_fp16=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# rerank_model_path = "/home/dev/med_search/BAAI/bge-reranker-v2-gemma"
rerank_model_path = "BAAI/bge-reranker-v2-gemma"
reranker = FlagLLMReranker(rerank_model_path, use_fp16=True)


@app.route('/get_embedding',methods=['POST'])
@api_key_required
def get_embedding():
    query = request.json.get('query')
    embeddings_ = model.encode([query],
                               batch_size=12,
                               max_length=8192,
                               # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                               )['dense_vecs']
    return json.dumps({'status':True,'content_vec':embeddings_.tolist()[0]})

@app.route('/get_token_length',methods=['POST'])
@api_key_required
def get_token_length():
    query = request.json.get('query')
    tokens = tokenizer.tokenize(query)
    return json.dumps({'status':True,'token_length':len(tokens)})

@app.route('/get_embedding_queries',methods=['POST'])
@api_key_required
def get_embedding_queries():
    query = request.json.get('query')
    embeddings_ = model.encode(query,
                               batch_size=12,
                               max_length=8192,
                               # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                               )['dense_vecs']
    return json.dumps({'status':True,'content_vec':embeddings_.tolist()})

# scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
# print(scores)
def clear_memory():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

@app.route('/get_rerank',methods=['POST'])
@api_key_required
def get_rerank():
    query_passages = request.json.get('query_passages')
    all_scores = []
    # print(query_passages)
    # scores = reranker.compute_score_single_gpu(query_passages,batch_size=16)
    # scores = reranker.compute_score(query_passages,batch_size=16)
    scores = reranker.compute_score(query_passages,batch_size=8)
    # scores2 = reranker2.compute_score(query_passages2,batch_size=8)
    clear_memory()
    return json.dumps({'status':True,'content_score':scores})

@app.route('/test_server',methods=['GET'])
def test_server():
    return json.dumps({'status':"server is running"})


if __name__=='__main__':
    app.run('127.0.0.1',port=443)
    # lenn = tokenizer.tokenize('测试一下具体什么效果')
    # print(tokenizer.convert_tokens_to_ids(lenn))
    # print(get_token_length())

