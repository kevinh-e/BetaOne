from huggingface_hub import snapshot_download

repo_id = "robertnurnberg/fishtest_pgns"
repo_type = "dataset"
allow_pattern = "25-03-04/*/*.pgn.gz"
local_dir = "./fishtest/"

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    allow_patterns=allow_pattern,
    local_dir=local_dir,
)
