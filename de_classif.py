import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse



'''
PART 1 : LOADING MODEL
'''


# ----- 1. Model Class -----
class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)  # Binary classification
    
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.pooler_output
        return torch.sigmoid(self.fc(cls)).squeeze()


#moved to main
'''
# ----- 2. Setup -----
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier().to(device)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
print("model loaded")
'''

###Not required anymore -> see predict_in_batches
'''
# ----- 3. Predict Function -----
def predict_sentences(sentences, model, tokenizer, device):
    model.eval()
    encodings = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    encodings.pop("token_type_ids", None)  # Optional if forward doesn't take it
    
    with torch.no_grad():
        probs = model(**encodings)  # Already sigmoid outputs from your forward
        preds = (probs >= 0.5).long()  # Threshold for binary classification
    
    return preds.cpu().numpy(), probs.cpu().numpy()
'''


'''
PART 2 : BATCHED INFERENCE SETUP
'''

# ----- 4. Dataset class for inference -----
class SentenceTermDataset(Dataset):
    def __init__(self, sentences, terms, tokenizer, max_length=128):
        self.sentences = sentences
        self.terms = terms
        self.encodings = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["term"] = self.terms[idx]
        item["sentence"] = self.sentences[idx]
        return item

# ----- 5. Batched prediction function -----
def predict_in_batches(sentences, terms, model, tokenizer, device, batch_size=32,threshold=0.5):
    dataset = SentenceTermDataset(sentences, terms, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    all_terms, all_sentences, all_preds, all_probs = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader):
            # Extract fields
            term_batch = batch.pop("term")
            sentence_batch = batch.pop("sentence")
            
            # Send tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("token_type_ids", None)  # Optional if model.forward doesn't take it

            # Forward pass
            probs = model(**batch)  # Already sigmoid outputs from your forward
            preds = (probs >= threshold).long() # Threshold goes HERE

            # Store
            all_terms.extend(term_batch)
            all_sentences.extend(sentence_batch)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_terms, all_sentences, all_preds, all_probs


#moved to main
'''
PART 3 : EXECUTION


matches = pd.read_csv("data/deft_matches.csv")
sentences = matches["sentence"].to_list()
terms = matches["terms"].to_list()


terms_out, sentences_out, preds, probs = predict_in_batches(
    sentences, terms, model, tokenizer, device, batch_size=16
)

# Save to CSV
df = pd.DataFrame({
    "term": terms_out,
    "sentence": sentences_out,
    "pred_label": preds,
    "pred_prob": [float(p) for p in probs]
})
df.to_csv("predictions.csv", index=False, sep=";")
'''



def main(input_path, output_path, model_path='base_de_bert.pt', batch_size=32, threshold=0.5):

    # ----- 2. model setup -----
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("model loaded")

    # ----- 6. execution -----
    matches = pd.read_csv(input_path)
    sentences = matches["sentence"].to_list()
    terms = matches["terms"].to_list()


    terms_out, sentences_out, preds, probs = predict_in_batches(
        sentences, terms, model, tokenizer, device, batch_size=batch_size,threshold=threshold
    )

    # ----- 7. save to csv -----
    df = pd.DataFrame({
        "term": terms_out,
        "sentence": sentences_out,
        "pred_label": preds,
        "pred_prob": [float(p) for p in probs]
    })
    df.to_csv(output_path, index=False, sep=";")
        
    #Warn user
    print ("result saved at "+output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sentences into definitional contexts")
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("output_path", help="Path to output file")
    
    parser.add_argument("--model_path", type=str, default="base_de_bert.pt", help="Path to model file (default = base_de_bert.pt)")
    parser.add_argument("--batch_size", type=int, default=32, help="Size of batches (default = 32)")
    parser.add_argument("--threshold", type=float, default = 0.5, help="threshold for label selection (default = 0.5)")
    args = parser.parse_args()

    main(**vars(args))

