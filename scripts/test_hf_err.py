import sys, traceback
from huggingface_hub import snapshot_download

try:
    print("Testing snapshot_download on smollm-corpus...")
    snapshot_download(
        repo_id='HuggingFaceTB/smollm-corpus',
        repo_type='dataset',
        allow_patterns=['python-edu/*', 'README.md'],
        local_dir=r'F:\Lingual_Data\raw\hf_permissive_stem_code\smollm-corpus-python-edu',
        max_workers=2
    )
    print("SUCCESS")
except Exception as e:
    print("EXCEPTION CAUGHT:")
    traceback.print_exc(file=sys.stdout)
