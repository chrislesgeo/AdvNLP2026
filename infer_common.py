import os
import torch
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def choose_device(cuda_flag: str = ""):
    if cuda_flag and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_test_file(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Test file not found: {path}")
    lines = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split('\t')
            # expect input\ttarget or input only
            lines.append(parts[0])
    return lines


def write_outputs(out_path: str, inputs: List[str], outputs: List[str]):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for inp, out in zip(inputs, outputs):
            f.write(inp.replace('\n', ' ') + '\t' + out.replace('\n', ' ') + '\n')
    logger.info(f"Wrote {len(outputs)} results to {out_path}")


def map_checkpoint_to_model_state(ckpt: Dict, model_state_keys: List[str]) -> Dict:
    """
    Attempt to map keys from a checkpoint dict to the model's state dict keys.
    This tries common prefix-stripping heuristics used in this repo (e.g. 'module.', 'module.model.').
    Returns a dict compatible with model.load_state_dict(..., strict=False).
    """
    mapped = {}
    prefixes = ["module.", "module.model.", "model.", "model.model.", "module.model.model."]
    for k, v in ckpt.items():
        # skip obvious non-weight entries
        if 'prompt' in k.lower() and 'embedding' in k.lower():
            continue
        if k.endswith('.pkl'):
            continue
        candidate = None
        # try raw key first
        if k in model_state_keys:
            candidate = k
        else:
            for p in prefixes:
                if k.startswith(p):
                    kk = k[len(p):]
                    if kk in model_state_keys:
                        candidate = kk
                        break
        if candidate is None:
            # also try removing a single leading component (for keys like "module.encoder.weight")
            kk = k.split('.', 1)[-1]
            if kk in model_state_keys:
                candidate = kk
        if candidate is not None:
            mapped[candidate] = v
    return mapped
