import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from torch.optim import AdamW
from tqdm import tqdm
import argparse

'''
CLASS PART
'''

# ----- dataset -----
class SentenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.encodings = tokenizer(df["Sentence"].tolist(), truncation=True, padding=True,
                                   max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(df["Label"].tolist()).float()

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)


# ----- Model -----
class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)  # Binary classification
    
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.pooler_output
        return torch.sigmoid(self.fc(cls)).squeeze()



'''
FUNCTION PART
'''

def evaluate(model,device,loader,loss_fn, return_loss=False):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch, labels in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = (outputs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    if return_loss:
        return acc, total_loss / len(loader)
    else:
        return acc
    


def predict_all(model,device,loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch, labels in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids, attention_mask)
            preds.extend((output > 0.5).int().cpu().numpy())
            targets.extend(labels.numpy())

    return targets, preds





#training and evaluation function
def training(datapath,model_path = "definitional_context_model.pt", epochs = 5, patience = 2, learning_rate=1e-5, weight_decay=0.01, verbose=True) :

    train_df = pd.read_csv(datapath+"task1_train.csv")
    dev_df = pd.read_csv(datapath+"task1_dev.csv")
    test_df = pd.read_csv(datapath+"task1_test_labeled.csv")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_data = SentenceDataset(train_df, tokenizer)
    dev_data   = SentenceDataset(dev_df, tokenizer)
    test_data  = SentenceDataset(test_df, tokenizer)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    dev_loader   = DataLoader(dev_data, batch_size=16)
    test_loader  = DataLoader(test_data, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    # ----- 5. Training -----
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        
        for batch, labels in tqdm(train_loader) :
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Collect predictions and labels for accuracy
            preds = (outputs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Compute training accuracy
        train_acc = accuracy_score(all_labels, all_preds)

        val_acc, val_loss = evaluate(model,device,dev_loader,loss_fn, return_loss=True)
        if (verbose) :
            print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            if (verbose) : print("new best model")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                if (verbose) : print("Early stopping.")
                break

        # Compute training accuracy
        train_acc = accuracy_score(all_labels, all_preds)


    # ----- 7. Run Eval -----
    model.load_state_dict(torch.load(model_path))

    y_true, y_pred = predict_all(model,device,test_loader)
    print(classification_report(y_true, y_pred, digits=4))



'''
MAIN PART
'''
def main(datapath,model_path = "definitional_context_model.pt", epochs = 5, patience = 2, learning_rate=1e-5, weight_decay=0.01, verbose=True):
    #all is contained in the training function
    training(datapath,model_path,epochs,patience, learning_rate,weight_decay,verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the post_processing classifier for termsuite")
    parser.add_argument("datapath", help="Path to the folder containing the task1_train, task1_dev and task1_test .csv")
    
    parser.add_argument("--model_path",  type=str, default = "definitional_context_model.pt", help="output path for the trained model")
    parser.add_argument("--epochs", type=int, default=5, help="number of epoch")
    parser.add_argument("--patience", type=int, default=2, help="early-stopping patience")
    parser.add_argument("--learning_rate", type=float, default = 1e-3, help="learning rate for training")
    parser.add_argument("--weight_decay", type=float, default = 0.01, help="weight decay for training")
    parser.add_argument("--verbose", type=bool, default = True, help="let the program print or not")
    
    args = parser.parse_args()

    main(**vars(args))
