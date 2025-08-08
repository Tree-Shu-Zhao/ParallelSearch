
file_path=/ephemeral

# --- For wiki-18 ---
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
port=8000

# --- For multihoprag ---
# index_file=$file_path/multihoprag-index/e5_Flat.index
# corpus_file=$file_path/multihoprag-index/corpus.jsonl
# port=8001

retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --port $port \
                                            --faiss_gpu