import pandas as pd
import argparse


def csv_to_html(input_csv, output_html):
    # Read CSV with semicolon separator
    df = pd.read_csv(input_csv, sep=";")
    
    # Lowercase terms
    df["term"] = df["term"].str.lower()
    
    # Split multiple terms into list (by comma or semicolon) and strip spaces
    df["term"] = df["term"].str.split(r"[;,]")
    df = df.explode("term")  # Duplicate rows for each term
    df["term"] = df["term"].str.strip()  # Remove leading/trailing spaces
    
    # Ensure correct types
    df["pred_prob"] = df["pred_prob"].astype(float)
    
    # Sort by probability (descending)
    df = df.sort_values(by="pred_prob", ascending=False)
    
    # Group by single term
    grouped = df.groupby("term")
    
    # Start HTML
    html = """
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h2 { color: #2c3e50; border-bottom: 2px solid #ccc; padding-bottom: 5px; }
            .sentence { margin-left: 20px; }
            .prob { color: #555; font-size: 0.9em; }
            .label-1 { background-color: #d4edda; padding: 3px 6px; border-radius: 4px; }
            .label-0 { background-color: #f8d7da; padding: 3px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
    <h1>Definition Extraction Results</h1>
    """

    # Add content
    for term, group in grouped:
        html += f"<h2>{term}</h2>\n"
        for _, row in group.iterrows():
            label_class = "label-1" if row["pred_label"] == 1 else "label-0"
            html += f"<div class='sentence'><b>Sentence:</b> {row['sentence']}<br>"
            html += f"<span class='prob'>Probability: {row['pred_prob']:.2f}</span> "
            html += f"<span class='{label_class}'>Label: {row['pred_label']}</span></div><br>\n"

    # End HTML
    html += "</body></html>"

    # Save file
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    
    #Warn user
    print ("html saved at "+output_html)


# Example usage
'''csv_to_html("regbert_predict_deft.csv", "regbert_predict_deft.html")'''


def main(input_path, output_path):
    csv_to_html(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert final CSV output as a HTML file")
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("output_path", help="Path to output file")
    args = parser.parse_args()

    main(**vars(args))
