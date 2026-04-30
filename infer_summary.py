import argparse
import os
from tqdm import tqdm
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from infer_common import choose_device, read_test_file, write_outputs, map_checkpoint_to_model_state


def load_model_and_tokenizer(model_name: str, device: torch.device):
    # choose family by name
    if 'pegasus' in model_name.lower():
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
    elif 't5' in model_name.lower() or model_name.lower().startswith('t5'):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif 'bart' in model_name.lower():
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    else:
        # fallback to Pegasus
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def try_load_checkpoint(ckpt_path: str, model):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Skipping checkpoint loading.")
        return 0
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # ckpt might be a dict with keys or a state_dict directly
    if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        raw = ckpt['state_dict']
    else:
        raw = ckpt
    model_keys = list(model.state_dict().keys())
    mapped = map_checkpoint_to_model_state(raw, model_keys)
    print(f"Mapped {len(mapped)} keys from checkpoint to model state (attempt).")
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"Loaded checkpoint into model. Missing keys: {len(missing)}; Unexpected: {len(unexpected)}")
    return len(mapped)


def generate_summaries(model, tokenizer, inputs, device, max_length=128, num_beams=4):
    outputs = []
    for inp in tqdm(inputs, desc="Generating"):
        batch = tokenizer([inp], truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            gen = model.generate(**batch, max_length=max_length, num_beams=num_beams)
        text = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        outputs.append(text)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='google/pegasus-large')
    parser.add_argument('--ckpt', default='pretrained_ckpt/019/bestckpt_full_model')
    parser.add_argument('--testfile', default='DATASETS/PromptSum/samsum/full_test.txt')
    parser.add_argument('--out', default='inference_summary_results.txt')
    parser.add_argument('--cuda', default='0')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_beams', type=int, default=4)
    args = parser.parse_args()

    device = choose_device(args.cuda)
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    # try to load checkpoint weights into the base model (heuristic mapping)
    try_load_checkpoint(args.ckpt, model)

    inputs = read_test_file(args.testfile)
    outputs = generate_summaries(model, tokenizer, inputs, device, max_length=args.max_length, num_beams=args.num_beams)

    write_outputs(args.out, inputs, outputs)


if __name__ == '__main__':
    main()
