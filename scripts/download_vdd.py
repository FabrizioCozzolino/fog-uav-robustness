from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id='RussRobin/VDD',
    repo_type='dataset',
    local_dir='data/raw/VDD',
)
print(f'Download completato in: {path}')
