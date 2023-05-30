class Config:
    llm_model_name = 'THUDM/chatglm-6b'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = 'GanymedeNil/text2vec-large-chinese'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = 'knowledge/faiss/'
    docs_path = 'knowledge/txt/'