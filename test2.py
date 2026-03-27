import os
import sys

# Reduce peak virtual-memory pressure when loading CUDA/cuDNN DLLs on Windows.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import torch, transformers, datasets, huggingface_hub, fsspec, spacy, catalogue
print(sys.version)
print('torch', torch.__version__)
print('cuda?', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('hub', huggingface_hub.__version__)
print('fsspec', fsspec.__version__)
print('spacy', spacy.__version__)
print('catalogue', getattr(catalogue, '__version__', 'unknown'))