import pandas as pd

n = 1000

df = pd.read_excel('SFBU_fine_tune_data.xlsx', sheet_name='Sheet1', 
        header=0, nrows=n)

# answers = df["Reason"].unique()

# reasons_dict = {reason: i for i, reason in enumerate(reasons)}
# df["prompt"] = "Drug: " + df["Drug_Name"] + "\n" + "Malady:"
# df["answer"] = " " + df["Reason"].apply(lambda x: "" + str(reasons_dict[x]))
df.rename(columns={"prompt": "prompt", "answer": "completion"}, inplace=True)
jsonl = df.to_json(orient="records", indent=0, lines=True)

with open("SFBU_fine_tune_data.jsonl", "w") as f:
    f.write(jsonl)
