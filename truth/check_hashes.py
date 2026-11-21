import hashlib
from pathlib import Path

data_dir = Path("D:/gci/inpuy")
for f in ["train.csv", "test.csv", "sample_submission.csv"]:
    p = data_dir / f
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest().upper()
        print(f"{f:22s} SHA-256: {h}")

