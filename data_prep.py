from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torchaudio
from sklearn.model_selection import train_test_split
data = []
save_path = "model_files"
for path in tqdm(Path("model_files/aesdd").glob("**/*.wav")):
    name = str(path).split('/')[-1].split('.')[0]
    label = str(path).split('/')[-2]

    try:
        # There are some broken files
        s = torchaudio.load(path)
        data.append({
            "name": name,
            "path": path,
            "emotion": label
        })
    except Exception as e:
        # print(str(path), e)
        pass
df = pd.DataFrame(data)
train_df, test_df_1 = train_test_split(df, test_size=0.3, random_state=101, stratify=df["emotion"])
test_df, dev_df = train_test_split(test_df_1, test_size=0.5, random_state=101, stratify=test_df_1["emotion"])

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
dev_df = dev_df.reset_index(drop=True)

train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
dev_df.to_csv(f"{save_path}/dev.csv", sep="\t", encoding="utf-8", index=False)


print(train_df.shape)
print(test_df.shape)
print(df.head())
# break