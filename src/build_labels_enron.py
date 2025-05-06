# build_labels_enron.py  (located in src/)
import pandas as pd
from pathlib import Path

root_data = Path(__file__).resolve().parents[1] / "data"
root_data.mkdir(exist_ok=True)

df = pd.read_csv(root_data / "enron_labeled.csv")

min_per_class = 3_000     # for 5 classes → 15 000 rows
df_bal = (df.groupby("category_label", group_keys=False)
            .sample(n=min_per_class, replace=True, random_state=1))

print(df_bal["category_label"].value_counts())

out_file = root_data / "enron_balanced.csv"
df_bal.to_csv(out_file, index=False)
print("Saved balanced file →", out_file)
