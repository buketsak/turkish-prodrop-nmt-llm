import pandas as pd
import ollama
import time

client = ollama.Client()

def translate_sentence(sentence):
    prompt = (
        f"Translate the following English sentence into Turkish:\n{sentence}\n"
        "Produce only the Turkish translation, without any additional explanations."
    )
    response = client.generate(model="translategemma:4b", prompt=prompt)
    return response.response.strip()

def translate_dataframe(df, source_col, target_col):
    translations = []
    for i, sentence in enumerate(df[source_col]):
        if pd.isna(sentence):
            translations.append("")
            continue
        try:
            translations.append(translate_sentence(sentence))
        except Exception as e:
            print(f"Error at row {i}: {e}")
            translations.append("")
        time.sleep(0.5)  # avoid memory spikes
    df[target_col] = translations
    return df


if __name__ == "__main__":

    df2 = pd.read_csv("data_nmt_2_all.csv")
    df2 = df2.drop(columns=["model4"])
    df2 = translate_dataframe(
        df2,
        source_col="source_sentence",
        target_col="translategemma-4b"
    )

    df2.to_csv("data2_nmt_2_all2.csv", index=False)

    print("Translation finished")
