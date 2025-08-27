import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import argparse
import re
#from tqdm import tqdm


'''
CLASS DEFINITION
'''

class HybridClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)



'''
FUNCTION DEFINITION (dataset preparation & run)
'''

#Import and prepare TermSuite results
def dataset_prep(ts_path,seuil) :

    df = pd.read_csv(ts_path, sep="\t")
    dx = pd.DataFrame([row for i,row in df.iterrows() if row["freq"]>seuil ])

    #extract the term and pos_tag to be used independantly later
    terms = dx[["key"]]
    splited_terms = []
    splited_pos = []
    for t in terms["key"] :
        splitrez=re.split(": ", t)
        splited_pos.append(splitrez[0])
        splited_terms.append(splitrez[1])
    dx["key_processed"]=splited_terms
    dx["pos"]=splited_pos

    return (dx)

#Helper to load the trained model
def load_model(model_path, input_dim, device="cpu"):
    model = HybridClassifier(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


#Batched inference function
def run_inference(dx, model, best_threshold, batch_size=32, device="cpu"):
    """
    Batched inference function.
    dx: dataframe with keys + numeric features
    model: loaded torch model
    best_threshold: threshold for classification
    batch_size: size of inference batch
    """
    # === 1. Prepare Data ===
    dx2 = dx[["key", "dFreq", "spec", "freq", "iFreq", "ind", "tfIdf", "swtSize"]]
    text_terms = dx2["key"].tolist()
    numeric_features = dx2[["dFreq", "spec", "freq", "iFreq", "ind", "tfIdf", "swtSize"]].values

    # Use the *same* scaler as training (here refitted each time – see note below)
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features)

    # Text embeddings
    bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    with torch.no_grad():
        text_embeddings = bert_model.encode(text_terms, convert_to_tensor=True, device=device)

    # Combine embeddings + numeric
    X_tensor = torch.cat(
        (text_embeddings, torch.tensor(numeric_features_scaled, dtype=torch.float32, device=device)),
        dim=1
    )

    # === 2. DataLoader ===
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)

    #print('starting inference')

    # === 3. Inference ===
    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            probs = model(xb).cpu().numpy().flatten()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    preds = (all_probs > best_threshold).astype(int)

    return all_probs, preds



'''
EXEMPLE USAGE :

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("model1_p0937_r0876_th0143.pt", input_dim=384+7, device=device)  
# 384 = MiniLM embedding dim, +7 numeric features

probs, preds = run_inference(dx, model, best_threshold=0.5, batch_size=128, device=device)

'''

#Export results to csv
def export_to_csv(dx,probs, preds, output_path, keep_noise=False):

    results = pd.DataFrame()
    results["term"] = dx["key_processed"].to_list()
    results["prob"]=probs
    results["preds"] = preds

    if keep_noise is True:
        results.to_csv(output_path,sep=";")
    else :
        res_purged = results[results['preds'] == 1.0].reset_index(drop=True)
        res_purged.to_csv(output_path,sep=";")
    
    #Warn user
    print(f"Found {sum(preds)} good terms. Results saved to {output_path}")




def main(input_path, output_path, model_path='base_classif_relu.pt', seuil_freq=5, batch_size=32, threshold=0.143,keep_noise=False):

    #data preparation
    dx=dataset_prep(input_path,seuil_freq)
    
    #inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, input_dim=384+7, device=device)  # 384 = MiniLM embedding dim, +7 numeric features
    probs, preds = run_inference(dx, model, best_threshold=threshold, batch_size=batch_size, device=device)
    
    #export
    export_to_csv(dx, probs, preds, output_path, keep_noise)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sentences into definitional contexts")
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("output_path", help="Path to output file")
    
    parser.add_argument("--model_path", type=str, default="base_classif_relu.pt", help="Path to model file (default = base_classif_relu.pt)")
    parser.add_argument("--seuil_freq", type=int, default = 5, help="threshold for term frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of batches (default = 32)")
    parser.add_argument("--threshold", type=float, default = 0.143, help="threshold for label selection (default = 0.143)")
    parser.add_argument("--keep_noise", type=bool, default = False, help="Keep noise in final result (default = False)")
    args = parser.parse_args()

    main(**vars(args))
