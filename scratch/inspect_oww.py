import openwakeword
from openwakeword.model import Model
import numpy as np

m = Model()
print(f"Model attributes: {dir(m)}")
if hasattr(m, "preprocessor"):
    print(f"Preprocessor attributes: {dir(m.preprocessor)}")
else:
    print("No preprocessor attribute found.")

# Try a prediction
m.predict(np.zeros(1280, dtype=np.float32))
if hasattr(m, "preprocessor"):
    if hasattr(m.preprocessor, "embeddings"):
        print("Embeddings found after predict.")
    else:
        print("No embeddings attribute on preprocessor.")
else:
    print("No preprocessor to check for embeddings.")
