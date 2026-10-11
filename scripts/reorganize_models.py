import os, shutil

def reorganize():
    models_dir = 'F:/Models'
    print(f'Starting safe reorganization of {models_dir}...')
    
    # 1. Meta-Llama-3-8B-Instruct
    llama_dir = os.path.join(models_dir, 'Meta-Llama-3-8B-Instruct')
    os.makedirs(llama_dir, exist_ok=True)
    for i in range(1, 5):
        src = os.path.join(models_dir, f'Meta-Llama-3-8B-Instruct-model-0000{i}-of-00004.safetensors')
        dst = os.path.join(llama_dir, f'model-0000{i}-of-00004.safetensors')
        if os.path.exists(src) and not os.path.exists(dst):
            print(f'Moving {src} -> {dst}')
            shutil.move(src, dst)
    src_idx = os.path.join(models_dir, 'Meta-Llama-3-8B-Instruct-model.safetensors.index.json')
    dst_idx = os.path.join(llama_dir, 'model.safetensors.index.json')
    if os.path.exists(src_idx):
        if not os.path.exists(dst_idx):
            shutil.move(src_idx, dst_idx)
        else:
            os.remove(src_idx)

    # 2. Qwen3.5-9B
    qwen_dir = os.path.join(models_dir, 'Qwen3.5-9B')
    os.makedirs(qwen_dir, exist_ok=True)
    src_s4 = os.path.join(models_dir, 'Qwen3.5-9B-model.safetensors-00004-of-00004.safetensors')
    dst_s4 = os.path.join(qwen_dir, 'model.safetensors-00004-of-00004.safetensors')
    if os.path.exists(src_s4) and not os.path.exists(dst_s4):
        print(f'Moving {src_s4} -> {dst_s4}')
        shutil.move(src_s4, dst_s4)

    # 3. llama-embed-nemotron-8b
    nemo_dir = os.path.join(models_dir, 'llama-embed-nemotron-8b')
    os.makedirs(nemo_dir, exist_ok=True)
    for i in range(1, 5):
        src = os.path.join(models_dir, f'llama-embed-nemotron-8b-model-0000{i}-of-00004.safetensors')
        dst = os.path.join(nemo_dir, f'model-0000{i}-of-00004.safetensors')
        if os.path.exists(src) and not os.path.exists(dst):
            print(f'Moving {src} -> {dst}')
            shutil.move(src, dst)
    src_nemo_idx = os.path.join(models_dir, 'llama-embed-nemotron-8b-model.safetensors.index.json')
    dst_nemo_idx = os.path.join(nemo_dir, 'model.safetensors.index.json')
    if os.path.exists(src_nemo_idx) and not os.path.exists(dst_nemo_idx):
        shutil.move(src_nemo_idx, dst_nemo_idx)

    # 4. Qwen3-Embedding-8B-model
    qemb_dir = os.path.join(models_dir, 'Qwen3-Embedding-8B-model')
    if os.path.exists(qemb_dir):
        for i in range(1, 5):
            src = os.path.join(qemb_dir, f'Qwen3-Embedding-8B-model-0000{i}-of-00004.safetensors')
            dst = os.path.join(qemb_dir, f'model-0000{i}-of-00004.safetensors')
            if os.path.exists(src) and not os.path.exists(dst):
                print(f'Renaming {src} -> {dst}')
                os.rename(src, dst)
        src_idx = os.path.join(qemb_dir, 'Qwen3-Embedding-8B-model.safetensors.index.json')
        dst_idx = os.path.join(qemb_dir, 'model.safetensors.index.json')
        if os.path.exists(src_idx) and not os.path.exists(dst_idx):
            os.rename(src_idx, dst_idx)

    # 5. Qwen3-14B-Uncensored
    q14_dir = os.path.join(models_dir, 'Qwen3-14B-Uncensored')
    if os.path.exists(q14_dir):
        for i in [2, 3]:
            src = os.path.join(q14_dir, f'Qwen3-14B-Uncensored-model-0000{i}-of-00006.safetensors')
            dst = os.path.join(q14_dir, f'model-0000{i}-of-00006.safetensors')
            if os.path.exists(src) and not os.path.exists(dst):
                print(f'Renaming {src} -> {dst}')
                os.rename(src, dst)
        src_idx = os.path.join(q14_dir, 'Qwen3-14B-Uncensored-model.safetensors.index.json')
        dst_idx = os.path.join(q14_dir, 'model.safetensors.index.json')
        if os.path.exists(src_idx) and not os.path.exists(dst_idx):
            os.rename(src_idx, dst_idx)

    # 6. bge-small-en-v1.5
    bge_dir = os.path.join(models_dir, 'bge-small-en-v1.5')
    os.makedirs(bge_dir, exist_ok=True)
    for fn in ['bge-small-en-v1.5-model.safetensors', 'bge-small-en-v1.5-pytorch_model.bin']:
        src = os.path.join(models_dir, fn)
        dst = os.path.join(bge_dir, fn.replace('bge-small-en-v1.5-', ''))
        if os.path.exists(src) and not os.path.exists(dst):
            print(f'Moving {src} -> {dst}')
            shutil.move(src, dst)

    # 7. nomic-embed-text-v1
    nomic_dir = os.path.join(models_dir, 'nomic-embed-text-v1')
    os.makedirs(nomic_dir, exist_ok=True)
    src_nomic = os.path.join(models_dir, 'nomic-embed-text-v1.safetensors')
    dst_nomic = os.path.join(nomic_dir, 'model.safetensors')
    if os.path.exists(src_nomic) and not os.path.exists(dst_nomic):
        print(f'Moving {src_nomic} -> {dst_nomic}')
        shutil.move(src_nomic, dst_nomic)

    # 8. Phi-3-mini-128k-instruct
    phi_dir = os.path.join(models_dir, 'Phi-3-mini-128k-instruct')
    os.makedirs(phi_dir, exist_ok=True)
    src_phi = os.path.join(models_dir, 'Phi-3-mini-128k-instruct.safetensors')
    dst_phi = os.path.join(phi_dir, 'model.safetensors')
    if os.path.exists(src_phi) and not os.path.exists(dst_phi):
        print(f'Moving {src_phi} -> {dst_phi}')
        shutil.move(src_phi, dst_phi)

    # 9. gemma-4-12B
    gemma_dir = os.path.join(models_dir, 'gemma-4-12B')
    if os.path.exists(gemma_dir):
        src_gemma = os.path.join(gemma_dir, 'gemma-4-12B.safetensors')
        dst_gemma = os.path.join(gemma_dir, 'model.safetensors')
        if os.path.exists(src_gemma) and not os.path.exists(dst_gemma):
            print(f'Renaming {src_gemma} -> {dst_gemma}')
            os.rename(src_gemma, dst_gemma)

    # 10. duplicates backup for root leftover files
    dup_dir = os.path.join(models_dir, '_duplicates_backup')
    os.makedirs(dup_dir, exist_ok=True)
    for fn in ['Qwen3.5-9B-model.safetensors-00001-of-00004.safetensors',
               'Qwen3.5-9B-model.safetensors-00002-of-00004.safetensors',
               'Qwen3.5-9B-model.safetensors-00003-of-00004.safetensors',
               'Qwen3.5-9B-model.safetensors.index.json']:
        src = os.path.join(models_dir, fn)
        dst = os.path.join(dup_dir, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f'Archiving duplicate {src} -> {dst}')
            shutil.move(src, dst)

    print('Reorganization complete.')

if __name__ == '__main__':
    reorganize()
