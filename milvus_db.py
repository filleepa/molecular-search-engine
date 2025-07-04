from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility

# create Milvus collection
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        return Collection(name=collection_name)
        
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, description="ids", is_primary=True, auto_id=False),
        FieldSchema(name="smiles", dtype=DataType.VARCHAR, description="SMILES", max_length=500),
        FieldSchema(name="embedding", dtype=DataType.BINARY_VECTOR, description="embedding vectors", dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description="molecular similarity search")
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        "index_type":"BIN_FLAT",
        "params":{"nlist":1024},
        "metric_type":"JACCARD"
    }
    
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection

collection = create_milvus_collection("molecular_search", 2048)

