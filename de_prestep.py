import os
import csv
import re
import nltk
from tqdm import tqdm
import pandas as pd
import argparse



def term_alignment(input_path, text_folder, output_path) :
    # ===== CONFIG =====
    TERMS_FILE = input_path     # CSV with a 'term' column OR single column of terms
    TEXT_FOLDER = text_folder       # Folder with .txt files
    OUTPUT_FILE = output_path       # Output CSV
    # ==================

    # --- Load terms with pandas ---
    df_terms = pd.read_csv(TERMS_FILE, sep=';')
    terms = df_terms['term'].dropna().astype(str).str.strip().tolist()

    # --- Build one big regex pattern ---
    # Sort terms by length so longer matches are preferred
    terms_sorted = sorted(set(terms), key=len, reverse=True)
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, terms_sorted)) + r')\b', re.IGNORECASE)

    # --- Sentence splitting ---
    def split_sentences(text):
        return nltk.sent_tokenize(text)

    # --- Process ---
    matches = []
    for filename in tqdm(os.listdir(TEXT_FOLDER)):
        if filename.lower().endswith(".txt") or filename.lower().endswith(".deft"):
            filepath = os.path.join(TEXT_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            for sentence in split_sentences(text):
                found_terms = set(re.findall(pattern, sentence))
                if found_terms:
                    matches.append({
                        "terms": sorted(found_terms),
                        "sentence": sentence.strip(),
                        "source_file": filename
                    })



    # --- Save ---
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["terms", "sentence", "source_file"])
        writer.writeheader()
        for row in matches:
            # Store the list of terms as comma-separated string
            writer.writerow({
                "terms": ', '.join(row["terms"]),
                "sentence": row["sentence"],
                "source_file": row["source_file"]
            })

    print(f"Found {len(matches)} matching sentences. Saved to {OUTPUT_FILE}")


# ------ MAIN ------


def main(term_file, text_folder, output_path):
    term_alignment(term_file, text_folder, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert final CSV output as a HTML file")
    parser.add_argument("term_file", help="Path to term csv file")
    parser.add_argument("text_folder", help="path to corpus folder")
    parser.add_argument("output_path", help="Path to output file")
    args = parser.parse_args()

    main(**vars(args))
