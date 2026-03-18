from huggingface_hub import snapshot_download

# Nome corretto del dataset su HF Hub
dataset_name = "databricks/databricks-dolly-15k"
dataset_dir = "./datasets/databricks-dolly-15k"

print(f"Scaricando {dataset_name} in {dataset_dir}...")
snapshot_download(
    repo_id=dataset_name,
    local_dir=dataset_dir,
    local_dir_use_symlinks=False,
    repo_type="dataset"
)
print("Dataset scaricato con successo!")
