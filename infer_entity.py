import argparse
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from infer_common import choose_device, read_test_file, write_outputs, map_checkpoint_to_model_state


def try_load_t5_checkpoint(ckpt_path: str, model):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Skipping T5 checkpoint loading.")
        return 0
    print(f"Loading checkpoint for T5: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        raw = ckpt['state_dict']
    else:
        raw = ckpt
    model_keys = list(model.state_dict().keys())
    mapped = map_checkpoint_to_model_state(raw, model_keys)
    print(f"Mapped {len(mapped)} keys from checkpoint to T5 model state (attempt).")
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"Loaded checkpoint into T5 model. Missing keys: {len(missing)}; Unexpected: {len(unexpected)}")
    return len(mapped)


def generate_entity_chains_with_t5(model, tokenizer, inputs, device, max_length=64):
    outputs = []
    for inp in inputs:
        batch = tokenizer([inp], truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            gen = model.generate(**batch, max_length=max_length, num_beams=4)
        text = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        outputs.append(text)
    return outputs


def generate_entity_chains_with_spacy(inputs):
    try:
        import spacy
    except Exception as e:
        raise RuntimeError('spacy is required for fallback entity extraction but is not installed')
    nlp = spacy.load('en_core_web_sm')
    outputs = []
    for inp in inputs:
        doc = nlp(inp)
        ents = [ent.text for ent in doc.ents]
        outputs.append(','.join(ents) if ents else 'none')
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='pretrained_ckpt/019/bestckpt_full_model')
    parser.add_argument('--testfile', default='DATASETS/PromptSum/samsum/full_test.txt')
    parser.add_argument('--out', default='inference_entity_results.txt')
    parser.add_argument('--cuda', default='0')
    parser.add_argument('--model_name', default='t5-base')
    args = parser.parse_args()

    device = choose_device(args.cuda)
    inputs = read_test_file(args.testfile)

    # try to load T5 tagger from checkpoint
    try:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        model.to(device)
        mapped = try_load_t5_checkpoint(args.ckpt, model)
        if mapped > 0:
            print('Using T5 tagger for entity inference')
            outputs = generate_entity_chains_with_t5(model, tokenizer, inputs, device)
            write_outputs(args.out, inputs, outputs)
            return
    except Exception as e:
        print('T5 tagger load failed or not present, falling back to spaCy:', e)

    # fallback to spaCy NER
    print('Falling back to spaCy NER for entity chains')
    outputs = generate_entity_chains_with_spacy(inputs)
    write_outputs(args.out, inputs, outputs)


if __name__ == '__main__':
    main()
