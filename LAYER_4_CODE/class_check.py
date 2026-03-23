import pandas as pd
cases = pd.read_csv("layer3_training_cases.csv")

print("Total cases:", len(cases))
print("High risk:", cases["label"].sum())
print("Low risk:", len(cases) - cases["label"].sum())

import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

