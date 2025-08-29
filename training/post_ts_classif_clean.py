import pandas as pd
#import os
#import time
import re
import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, roc_curve #,precision_score, recall_score, f1_score
#import matplotlib.pyplot as plt
import argparse
from pathlib import Path




'''
CLASS PART
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


class TermDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    '''def __getitem__(self, idx):
        return self.X[idx], self.y[idx]'''
    def __getitem__(self, idx):
        return self.X[idx].float(), self.y[idx].float()


'''
FUNCTION PART
'''

#get the training dataset ready
def get_data(folder_ts_path,folder_termino_path,seuil=1,verbose=True) :
    
    if (verbose) :
        print("Starting data extraction")

    #gather the datasets from the folders
    folder = Path(folder_ts_path)
    all_files = list(folder.glob("*.tsv")) + list(folder.glob("*.csv"))

    if not all_files:
        raise FileNotFoundError(f"No .tsv files found in {folder_ts_path}")

    dfs = [pd.read_csv(file, sep='\t', on_bad_lines='skip') for file in all_files]
    df = pd.concat(dfs, ignore_index=True)


    folder_gold = Path(folder_termino_path)
    all_files = list(folder_gold.glob("*.tsv")) + list(folder_gold.glob("*.csv"))

    if not all_files:
        raise FileNotFoundError(f"No .csv files found in {folder_termino_path}")

    dgs = [pd.read_csv(file, sep=';', on_bad_lines='skip') for file in all_files]
    dg = pd.concat(dgs, ignore_index=True)

    #processing seuil de fréquence
    df = pd.DataFrame([row for _,row in df.iterrows() if row["freq"]>seuil ])


    #process the gold_terms

    gold_terms = dg[["prefLabel_en","altLabel_en","hiddenLabel_en"]]

    hidden1=gold_terms[["hiddenLabel_en"]]
    alternative1 = gold_terms[["altLabel_en"]]
    prefered1 = gold_terms["prefLabel_en"]

    gold = []
    gold.extend(prefered1)

    for text in hidden1["hiddenLabel_en"] :
        x = re.split("§§", str(text))
        if x != ["nan"] :
            gold.extend(x)

    for text in alternative1["altLabel_en"] :
        x = re.split("§§", str(text))
        if x != ["nan"] :
            gold.extend(x)

    #séparation terme / POS tag (nécessaire pour projection du gold)
    terms = df[["key"]]
    splited_terms = []
    splited_pos = []

    for t in terms["key"] :
        splitrez=re.split(": ", t)
        splited_pos.append(splitrez[0])
        splited_terms.append(splitrez[1])

    df["key_processed"]=splited_terms
    df["key_processed"]

    df["pos"]=splited_pos


    #projection de la terminologie sur l'extraction (def gold)
    df["in_gold"] = df["key_processed"].isin(gold).astype(int)

    #warn user if verbose
    if (verbose) :
        print("Data extraction complete")

    #retour
    return df




#Training function
def training(df, epoch=50,learning_rate=1e-3,verbose=True,path_output_model="post_pro_model.pt") :

    if (verbose) :
        print("Starting training function")

    dx = df[["key", "dFreq", "spec","freq","iFreq","ind","tfIdf","swtSize","in_gold"]]

    text_terms = dx['key'].tolist()

    numeric_features = dx[["dFreq", "spec","freq","iFreq","ind","tfIdf","swtSize"]].values
    labels = dx['in_gold'].values

    # === 2. Preprocess features ===
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features)

    #print(np.isnan(numeric_features_scaled).any())   # True = bad
    #print(np.isinf(numeric_features_scaled).any())   # True = bad

    # === 3. Generate BERT embeddings ===
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    with torch.no_grad():
        text_embeddings = bert_model.encode(text_terms, convert_to_tensor=True)

    # === 4. Combine text and numeric features ===
    text_tensor = text_embeddings  # shape [N, 384]
    numeric_tensor = torch.tensor(numeric_features_scaled, dtype=torch.float32)  # shape [N, F]

    # Final input shape: [N, 384 + F]
    X = torch.cat((text_tensor, numeric_tensor), dim=1)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # [N, 1]

    # === 5. Dataset & Dataloader ===
    dataset = TermDataset(X, y) #Added


    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16)
    test_dl = DataLoader(test_ds, batch_size=16)

    # === 6. Feedforward Classifier ===
    model = HybridClassifier(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    # === 7. Training Loop ===
    for epoch in range(epoch):
        # === Training ===
        model.train()
        total_loss = 0
        for xb, yb in tqdm(train_dl):
            preds = model(xb)
            #print(preds.min().item(), preds.max().item())  # Should be between 0 and 1
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # === Validation ===
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_dl):
                preds = model(xb)
                loss = loss_fn(preds, yb)
                total_val_loss += loss.item()
                pred_labels = (preds > 0.5).float()
                correct += (pred_labels == yb).float().sum().item()
                total += yb.size(0)

        val_accuracy = correct / total
        if(verbose) :
            print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss:.4f} | Val Loss: {total_val_loss:.4f} | Val Acc: {val_accuracy:.2%}")

    #Save model
    torch.save(model.state_dict(), path_output_model)
    #return model and test_dataloader
    return (model, test_dl)

def evaluate_model(model, dataloader, name="Set"):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in tqdm(dataloader):
            probs = model(xb).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(yb.cpu().numpy().flatten())
    preds = (np.array(all_probs) > 0.5).astype(int) # THRESHOLD ?
    print(f"\n📊 Classification Report ({name}):")
    print(classification_report(all_labels, preds, digits=3))
    print("🧮 Confusion Matrix:")
    print(confusion_matrix(all_labels, preds))
    auc = roc_auc_score(all_labels, all_probs)
    print(f"🔵 ROC AUC: {auc:.3f}")
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    '''
    #Plotting roc auc curve

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curve ({name})")
    plt.legend(); plt.grid(True)
    plt.show()
    '''




def main(folder_ts_path,folder_termino_path,seuil=1,epoch=50,learning_rate=1e-3,verbose=True,path_output_model="post_pro_model.pt"):

    #data :
    df = get_data(folder_ts_path,folder_termino_path,seuil)

    #training
    model,test_dl = training(df,epoch,learning_rate,verbose,path_output_model)
    
    #evaluation
    evaluate_model(model,test_dl,path_output_model)

    #Warn user
    if (verbose) :
        print ("model saved at "+path_output_model)

#folder_ts_path,folder_termino_path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the post_processing classifier for termsuite")
    parser.add_argument("folder_ts_path", help="Path to the folder containing the termsuite extraction files")
    parser.add_argument("folder_termino_path", help="Path to the folder containing the terminologies")

    parser.add_argument("--seuil", type=int, default=1, help="frequency threshold for the Termsuite extractions)")
    parser.add_argument("--epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--learning_rate", type=float, default = 1e-3, help="learning rate for training")
    parser.add_argument("--verbose", type=bool, default = True, help="let the program print or not")
    parser.add_argument("--path_output_model", type=str, default = "post_pro_model.pt", help="output path for the trained model")
    
    args = parser.parse_args()

    main(**vars(args))
