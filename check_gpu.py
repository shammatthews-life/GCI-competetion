import torch
import lightgbm
import catboost

print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
print("LightGBM version:", lightgbm.__version__)
print("CatBoost version:", catboost.__version__)

