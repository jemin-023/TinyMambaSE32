import os
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_dir = "speech-noise-dataset"
clean_dir = os.path.join(dataset_dir,"clean_speech")
noisy_dir = os.path.join(dataset_dir,"noisy_speech")

seed = 18

pairs = []
for fname in sorted(os.listdir(clean_dir)):
    if(fname.endswith(".wav")) and os.path.exists(os.path.join(noisy_dir,fname)):
        pairs.append({
            "file_id": fname,
            "noisy_path": os.path.join(noisy_dir, fname),
            "clean_path": os.path.join(clean_dir, fname)
        })
    
df = pd.DataFrame(pairs)


# Split: 80% Train, 10% Val, 10% Test
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=seed, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, shuffle=True)

train_df.to_csv(os.path.join(dataset_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(dataset_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")