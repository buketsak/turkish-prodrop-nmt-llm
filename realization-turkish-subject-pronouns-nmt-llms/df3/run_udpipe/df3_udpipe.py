import pandas as pd
import re
import requests
import pyconll
import json

df = pd.read_csv("data_nmt_3_all2.csv")
df = df.drop(columns=["reviewer", "reviewer_notes"])

PRONOUNS = {"ben", "sen", "o", "biz", "siz", "onlar"} #"o" The forms "oysa" do not exist in standard Turkish. "o ise" = as for him/her/it Göksel, Kerslake (2005)p.102
POSSESSIVES = {"onunki", "seninki", "bizinki", "sizinki", "onlarınki"}

def normalize_clitics(text):
    #pronouns + se/sa
    pattern1 = re.compile(r'\b(' + '|'.join(PRONOUNS) + r')(se|sa)\b', flags=re.IGNORECASE)
    text = pattern1.sub(r'\1 ise', text)
    #possessives + yse/ysa
    pattern2 = re.compile(r'\b(' + '|'.join(POSSESSIVES) + r')(yse|ysa)\b', flags=re.IGNORECASE)
    text = pattern2.sub(r'\1 ise', text)
    return text

df = df.rename(columns={
    "model1": "opus-tatoeba-en-tr",
    "model2": "m2m100-418M",
    "model3": "NLLB200-3.3B"
})

model_cols = [
    "human_translations",
    "opus-tatoeba-en-tr",
    "m2m100-418M",
    "NLLB200-3.3B",
    "translategemma-4b",
    "llama3.1-IT-8B"
]

for col in model_cols:
    df[col] = df[col].astype(str).apply(normalize_clitics)
    
#initialize the columns to hold lists
for col in model_cols:
    df[f"subject_pro_{col}"] = [[] for _ in range(len(df))]
    df[f"pro_labels_{col}"] = [[] for _ in range(len(df))]

def parse_sentences(sentences_chunk):
    """Send a list of sentences to UDPipe REST server and get CoNLL-U output"""
    text = "\n".join(sentences_chunk)
    url = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
    data = {
        "model": "turkish-kenet-ud-2.17-251125",
        "data": text,
        "tokenizer": "horizontal", # raw text tokenizer, each sentence on a separate line, with tokens separated by spaces.
        "tagger": "1",             # enable POS tagging
        "parser": "1",             # enable dependency parsing
        "output": "conllu"         # output format
    }
    response = requests.post(url, data=data)
    try:
        parsed_json = response.json()
        return parsed_json.get("result", "")
    except:
        return response.text
    
# process in batches
BATCH_SIZE = 20

for col in model_cols:
    print(f"Processing: {col}")

    conllu_output_file = f"parsed_{col}.conllu"

    # create file
    with open(conllu_output_file, "w", encoding="utf-8") as f:
        pass #for now
    for start in range(0, len(df), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_sentences = df[col].iloc[start:end].tolist()

        # skip empty batches
        if not any(batch_sentences):
            continue
        # get conllu for the batch
        conllu_text = parse_sentences(batch_sentences)
    
        with open(conllu_output_file, "a", encoding="utf-8") as f:
            f.write(conllu_text)
            f.write("\n")  #newline between batches
    
        conllu = pyconll.load_from_string(conllu_text)

        # loop over sentences in batch
        for i, sent in enumerate(conllu):
            pronouns = []
            labels = []

            # find verbs that already have overt subjects
            verbs_with_subjects = {int(t.head) for t in sent if t.deprel.startswith("nsubj")}
            # overt subjects
            for t in sent:
                #overt
                if t.deprel.startswith("nsubj"):
                    pronouns.append(t.form.lower())
                    labels.append("1")

                # dropped pronouns: verbs without overt subjects
                elif t.upos == "VERB" and int(t.id) not in verbs_with_subjects:
                    pronouns.append("-")
                    labels.append("0")

                # limit to 2 subjects per sentence
                if len(pronouns) >= 2:
                    break        #limit to 2 pronouns per sentence, special for our data
            # ensure exactly 2 entries
            while len(pronouns) < 2:
                pronouns.append("-")
                labels.append("0")
    
            pronouns = pronouns[:2]
            labels = labels[:2]

            # assign to dataframe
            df.at[start + i, f"subject_pro_{col}"] = pronouns
            df.at[start + i, f"pro_labels_{col}"] = labels
        
for col in model_cols:
    df[f"subject_pro_{col}"] = df[f"subject_pro_{col}"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    df[f"pro_labels_{col}"] = df[f"pro_labels_{col}"].apply(lambda x: json.dumps(x, ensure_ascii=False))
df.to_csv("data3_nmt_analysis.csv", index=False, encoding="utf-8")

print(df[["subject_pro_m2m100-418M", "pro_labels_m2m100-418M"]].head(10))

df = pd.read_csv("data3_nmt_analysis.csv")

for col in model_cols:
    df[f"subject_pro_{col}"] = df[f"subject_pro_{col}"].apply(json.loads)
    df[f"pro_labels_{col}"] = df[f"pro_labels_{col}"].apply(json.loads)
