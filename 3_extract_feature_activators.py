import nltk
import torch
import scipy.stats
import pandas as pd
from sae_lens import SAE
from datasets import Dataset
from collections import Counter
from huggingface_hub import login
from sae_lens import ActivationsStore
from config import openai_token, hf_token
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

nltk.download('stopwords')

HF_TOKEN = hf_token
OPENAI_API_KEY = openai_token
login(token=HF_TOKEN)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
hook_points = model.hook_dict
tokenizer = model.tokenizer

sae, cfg_dict, _ = SAE.from_pretrained("darpanaswal/TER-SAEs", "blocks.0.hook_attn_out")
sae.to(device)
sae.fold_W_dec_norm()

tok = pd.read_csv("data/jailbreakTexts.csv")
tok = Dataset.from_pandas(tok)
token_dataset = tokenize_and_concatenate(
    dataset=tok,
    tokenizer=model.tokenizer,  
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)

all_tokens = torch.stack([x['tokens'] for x in token_dataset])

hook_point = "blocks.0.hook_attn_out"
prompts = tok['text']

activation_store = ActivationsStore.from_sae(
    model=model,
    sae=sae,
    streaming=True,
    store_batch_size_prompts=8,
    train_batch_size_tokens=4096,
    n_batches_in_buffer=32,
    device=device,
)

all_activations = []
token_info = []

for prompt_idx, prompt in enumerate(prompts):
    tokenized = model.tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    tokens = model.tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])

    _, cache = model.run_with_cache(prompt)
    activations = cache[hook_point].squeeze(0)

    # Sanity check
    if activations.shape[0] != len(tokens):
        print(f"❗ Warning: Token length ({len(tokens)}) and activation length ({activations.shape[0]}) mismatch!")
    else:
        print(f"✅ Token count: {len(tokens)} | Activation shape: {activations.shape}")

    all_activations.append(activations)
    token_info.extend([(token_text, prompt) for token_text in tokens])

# Stack activations into single tensor
all_activations = torch.cat(all_activations, dim=0)
print(f"\n📦 Total tokens collected: {all_activations.shape[0]}")
print(f"🧠 Running SAE encoding on shape: {all_activations.shape}\n")

# Encode and compute absolute activations
sparse_codes = sae.encode(all_activations)
sparse_activations = sparse_codes.abs()

# Identify top-k token indices per feature
top_k = 10
top_activating_indices = sparse_activations.topk(top_k, dim=0).indices

feature_data = []

for feature_id in range(sparse_codes.shape[1]):
    top_tokens = []
    top_prompts = []
    mean_activation = sparse_activations[:, feature_id].mean().item()

    for token_index in top_activating_indices[:, feature_id].cpu().numpy():
        token_text, prompt_text = token_info[token_index]
        top_tokens.append(token_text)
        top_prompts.append(prompt_text)

    token_counts = Counter(top_tokens)
    token_entropy = scipy.stats.entropy(list(token_counts.values()))

    feature_data.append({
        "feature_idx": feature_id,
        "activating_prompts": list(set(top_prompts)),
        "top_tokens": top_tokens,
        "mean_activation": mean_activation,
        "token_entropy": token_entropy,
    })

# Convert to DataFrame and save
feature_df = pd.DataFrame(feature_data)
feature_df.to_csv("data/feature_activations.csv", index=False)

print(f"🧮 Total features processed: {len(feature_df)}")