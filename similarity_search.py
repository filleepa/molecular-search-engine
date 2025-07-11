from towhee import ops, pipe, DataCollection
from milvus_db import MILVUS_HOST, MILVUS_PORT
from joblib import load
from rdkit import Chem
from rdkit.Chem import Draw
from towhee.types.image_utils import from_pil

def to_images(data):
    """Takes a SMILES input and converts it to a towhee.Image object for display."""
    imgs = []
    for smiles in data:
        mol = Chem.MolFromSmiles(smiles)
        img = from_pil(Draw.MolToImage(mol))
        imgs.append(img)
    return imgs

search_pipe = (pipe.input("query_smiles")
               .map("query_smiles", "fp", ops.molecular_fingerprinting.rdkit(algorithm="daylight"))
               .flat_map("fp", ("super_id", "super_score", "super_smiles"), 
                         ops.ann_search.milvus_client(host=MILVUS_HOST,
                                                      port=MILVUS_PORT,
                                                      collection_name="molecular_search",
                                                      limit=6,
                                                      param={"metric_type":"JACCARD", "nprobe":10},
                                                      output_fields=["smiles"]))
               .flat_map("fp", ("sub_id", "sub_score", "sub_smiles"),
                         ops.ann_search.milvus_client(host=MILVUS_HOST,
                                                      port=MILVUS_PORT,
                                                      collection_name="molecular_search",
                                                      limit=6,
                                                      param={"metric_type": "JACCARD", "nprobe": 10},
                                                      output_fields=["smiles"]))
               .window_all("query_smiles", "query_smiles", lambda x: to_images(x[:1]))
               .window_all("sub_smiles", "sub_smiles", to_images)
               .window_all("super_smiles", "super_smiles", to_images)
               .output("query_smiles", "sub_smiles", "super_smiles")
)

res = search_pipe("Cn1ccc(=O)nc1")
DataCollection(res).show()
