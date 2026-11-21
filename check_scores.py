import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    df = pd.read_csv(r"d:\gci\oof_predictions_v10_conservative_grade.csv")
    print("oof_predictions_v10_conservative_grade.csv:")
    print("Columns:", df.columns.tolist())
    for col in df.columns:
        if col not in ['SK_ID_CURR', 'TARGET', 'fold']:
            score = roc_auc_score(df['TARGET'], df[col])
            print(f"  {col}: {score:.6f}")
except Exception as e:
    print("Error:", e)

