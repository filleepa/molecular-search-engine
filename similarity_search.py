from towhee import ops, pipe, DataCollection
from milvus_db import MILVUS_HOST, MILVUS_PORT

search_pipe = (pipe.input("query_smiles")
               .map("query_smiles", "fp", ops.molecular_fingerprinting.rdkit(algorithm="daylight"))
               .flat_map("fp", ("id", "score", "similar_smiles"), ops.ann_search.milvus_client(host=MILVUS_HOST,
                                                                                               port=MILVUS_PORT,
                                                                                               collection_name="molecular_search",
                                                                                               param={"metric_type":"JACCARD", "nprobe":10},
                                                                                               output_fields=["smiles"]))
               .output("query_smiles", "similar_smiles")
)

res = search_pipe("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
DataCollection(res).show()
