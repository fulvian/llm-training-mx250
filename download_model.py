from huggingface_hub import snapshot_download

# Scarica SmolLM-135M-Instruct
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
local_dir = "./models/SmolLM-135M-Instruct"

print(f"Scaricando {model_name} in {local_dir}...")
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Modello scaricato con successo!")

# Scarica dataset databricks-dolly-15k
dataset_name = "databricks/databricks-dolly-15k"
dataset_dir = "./datasets/databricks-dolly-15k"

print(f"\nScaricando {dataset_name} in {dataset_dir}...")
snapshot_download(
    repo_id=dataset_name,
    local_dir=dataset_dir,
    local_dir_use_symlinks=False
)
print("Dataset scaricato con successo!")
