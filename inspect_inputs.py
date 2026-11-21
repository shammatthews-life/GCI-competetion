import pandas as pd

train = pd.read_csv(r"d:\gci\inpuy\train.csv", nrows=5)
test = pd.read_csv(r"d:\gci\inpuy\test.csv", nrows=5)
sample = pd.read_csv(r"d:\gci\inpuy\sample_submission.csv", nrows=5)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample shape:", sample.shape)
print("Target column in train:", "TARGET" in train.columns)
print("Sample head:\n", sample.head(2))

