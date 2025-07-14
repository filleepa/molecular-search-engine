from milvus_db import MILVUS_HOST, MILVUS_PORT
from towhee import pipe, ops
import gradio
from similarity_search import to_images

def search_smiles_with_metric(smiles, metric_type):
    search_func = (pipe.input("query_smiles")
                   .map("query_smiles", "fp", ops.molecular_fingerprinting.rdkit(algorithm="daylight"))
                   .flat_map("fp", ("id", "score", "similar_smiles"), ops.ann_search.milvus_client(host=MILVUS_HOST,
                                                                                                   port=MILVUS_PORT,
                                                                                                   collection_name="molecular_search",
                                                                                                   limit=5,
                                                                                                   param={"metric_type": metric_type, "nprobe":10},
                                                                                                   output_fields=["smiles"]))
                   .window_all("query_smiles", "query_smiles", lambda x: to_images(x[:1]))
                   .window_all("similar_smiles", "similar_smiles", to_images)
                   .output("similar_smiles")
                   )
    return search_func(smiles).to_list()[0][0]

interface = gradio.Interface(search_smiles_with_metric,
                             [gradio.components.Textbox(lines=1, inputs="CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
                              gradio.components.Radio(["JACCARD"])],
                             [gradio.components.Image(type="pil", label=None) for _ in range(5)],
                             live=True
                             )

interface.launch(inline=True, share=True)