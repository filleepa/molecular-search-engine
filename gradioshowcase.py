from milvus_db import MILVUS_HOST, MILVUS_PORT
from towhee import pipe, ops
import gradio as gr
from rdkit import Chem
from rdkit.Chem import Draw
from towhee.types.image_utils import from_pil

def to_images_with_score(data, score_list):
    """Takes a SMILES input and converts it to a towhee.Image object for display. Returns a list of PIL.Image, caption tuples"""
    imgs = []
    for smiles, score in zip(data, score_list):
        mol = Chem.MolFromSmiles(smiles)
        img = from_pil(Draw.MolToImage(mol))
        caption = f"{smiles}\nScore: {score:.3f}"
        imgs.append((img, caption))
    return imgs

def search_smiles_with_metric(smiles, metric_type):
    search_func = (pipe.input("query_smiles")
                   .map("query_smiles", "fp", ops.molecular_fingerprinting.rdkit(algorithm="daylight"))
                   .flat_map("fp", ("id", "score", "similar_smiles"), ops.ann_search.milvus_client(host=MILVUS_HOST,
                                                                                                   port=MILVUS_PORT,
                                                                                                   collection_name="molecular_search",
                                                                                                   limit=5,
                                                                                                   param={"metric_type": metric_type, "nprobe":10},
                                                                                                   output_fields=["smiles"]))
                   .window_all(("similar_smiles", "score"), "results", to_images_with_score)
                   .output("results")
                   )
    return search_func(smiles).to_list()[0][0]

interface = gr.Interface(fn=search_smiles_with_metric,
                             inputs=[gr.components.Textbox(lines=1, label="Query SMILES", value="CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
                              gr.components.Radio(["JACCARD"], label="Metric")],
                             outputs=gr.Gallery(label="Top 5 Similar Structures", columns=5, type="pil"),
                             live=True
                             )

if __name__ == "__main__":
    interface.launch(inline=True, share=True)