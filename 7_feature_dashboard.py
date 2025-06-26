import torch
import pandas as pd
from sae_lens import SAE
from config import hf_token
from datasets import Dataset
from huggingface_hub import login
from transformer_lens import HookedTransformer
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from transformer_lens.utils import tokenize_and_concatenate
from sae_dashboard.data_writing_fns import save_feature_centric_vis

HF_TOKEN = hf_token
login(token=HF_TOKEN)

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
hook_points = model.hook_dict
tokenizer = model.tokenizer

sae, cfg_dict, _ = SAE.from_pretrained("darpanaswal/TER-SAEs", "blocks.0.hook_attn_out")

tok = pd.read_csv("data/jailbreakTexts.csv")
tok = Dataset.from_pandas(tok)
token_dataset = tokenize_and_concatenate(
    dataset=tok,  # type: ignore
    tokenizer=model.tokenizer,  # type: ignore
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)

all_tokens = torch.stack([x['tokens'] for x in token_dataset])

# Configure visualization
config = SaeVisConfig(
    hook_point=sae.cfg.hook_name,
    features=list(range(100)),
    minibatch_size_features=64,
    minibatch_size_tokens=256,
    device="cuda:1",
    dtype="bfloat16"
)

data = SaeVisRunner(config).run(encoder=sae, model=model, tokens=all_tokens)
save_feature_centric_vis(sae_vis_data=data, filename="feature_dashboard.html")