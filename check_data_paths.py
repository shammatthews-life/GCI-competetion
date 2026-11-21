import os

for path in [r"d:\gci\inpuy", r"d:\gci\input", r"d:\gci"]:
    if os.path.exists(path):
        train = os.path.join(path, "train.csv")
        test = os.path.join(path, "test.csv")
        sample = os.path.join(path, "sample_submission.csv")
        print(f"Path {path}: train={os.path.exists(train)}, test={os.path.exists(test)}, sample={os.path.exists(sample)}")

