import time
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

X = pd.DataFrame(np.random.randn(30000, 20))
y = np.random.randint(0, 2, 30000)
t0 = time.time()
cb = CatBoostClassifier(iterations=50, depth=5, verbose=False, thread_count=4)
cb.fit(X, y)
print(f"50 iterations took: {time.time()-t0:.2f}s")

