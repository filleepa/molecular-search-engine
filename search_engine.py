from towhee import ops, pipe
from joblib import load
from milvus_db import MILVUS_HOST, MILVUS_PORT, collection
from pymilvus import Collection

df = load("pubchem_df.pkl")

insert_pipe = (pipe.input("df")
               .flat_map("df", ("smiles", "id"), 
                         lambda df: df.values.tolist()) # read tabular data from the df and flatten (smiles and id columns)
               .map("smiles", "fp", 
                    ops.molecular_fingerprinting.rdkit(algorithm="daylight")) # use the daylight algorithm to generate fingerprint with the rdkit operator in towhee hub
               .map(("id", "smiles", "fp"), "res", 
                    ops.ann_insert.milvus_client(host=MILVUS_HOST,
                                                port=MILVUS_PORT,
                                                collection_name="molecular_search")) # inserts molecular fingerprints into Milvus
               .map("smiles", "smiless", lambda x: print(x))
               .output("res")
               )

insert_pipe(df)
collection.flush()
collection.load()
print(f"Total number of inserted data is {collection.num_entities}.")
