import torch
from config import hf_token
from transformer_lens import HookedTransformer
from sae_lens import upload_saes_to_huggingface, LanguageModelSAERunnerConfig, SAETrainingRunner, CacheActivationsRunnerConfig, CacheActivationsRunner

HF_TOKEN = hf_token

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
hook_points = model.hook_dict
tokenizer = model.tokenizer

total_training_steps = 300
batch_size = 4096
total_training_tokens = total_training_steps*batch_size

cfg = LanguageModelSAERunnerConfig(
    architecture = "jumprelu",
    model_name = "meta-llama/Llama-3.2-1B-Instruct",
    model_class_name = "HookedTransformer",
    hook_name = "blocks.0.hook_attn_out",
    hook_layer = 0,
    dataset_path = "darpanaswal/TERInterpData",
    context_size = 128,
    d_in = 2048,
    b_dec_init_method = "zeros",
    expansion_factor = 4,
    # activation_fn = "jumprelu",
    apply_b_dec_to_input = False,
    dtype = "bfloat16",
    prepend_bos = True,
    log_to_wandb = False,
    normalize_activations = "none",
    l1_coefficient = 8e-5,
    lp_norm = 1.0,
    mse_loss_normalization = "batch_norm",
    device = device,
    training_tokens = total_training_tokens,

)
sae = SAETrainingRunner(cfg).run()

saes_dict = {
    "blocks.0.hook_attn_out": sae
}

upload_saes_to_huggingface(
    saes_dict,
    hf_repo_id="darpanaswal/TER-SAEs",
)