import os
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained_vae


def get_rvqvae_embeddings(vae, code_idx):
    """
    Get quantized embeddings from R-VQVAE token indices.
    Sums embeddings from all quantization stages (residual reconstruction).

    Args:
        vae: RVQVae model
        code_idx: Token indices [B, T', num_quantizers]

    Returns:
        embeddings: [B, T', code_dim]
    """
    B, T_prime, num_quantizers = code_idx.shape

    # Accumulate embeddings from all quantization stages
    embeddings = None
    for i in range(num_quantizers):
        indices = code_idx[:, :, i]  # [B, T']
        indices_flat = indices.reshape(-1)  # [B*T']

        # Get codebook embedding for this stage
        quantizer = vae.quantizer._get_quantizer(i)
        z_q = quantizer.dequantize(indices_flat)  # [B*T', code_dim]

        # Reshape to [B, T', code_dim]
        z_q = z_q.view(B, T_prime, vae.code_dim)

        # Sum residual embeddings
        if embeddings is None:
            embeddings = z_q
        else:
            embeddings = embeddings + z_q

    return embeddings.detach().cpu().numpy()


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")

    # Determine output paths
    tokens_code_path = cfg.DATASET.CODE_PATH

    # Create embeddings path (parallel structure)
    if 'TOKENS' in tokens_code_path.upper():
        embeddings_code_path = tokens_code_path.replace('TOKENS', 'EMBEDDINGS').replace('tokens', 'embeddings')
    else:
        tokens_code_path = os.path.join(tokens_code_path, 'TOKENS')
        embeddings_code_path = os.path.join(cfg.DATASET.CODE_PATH, 'EMBEDDINGS')

    print(f"\nOutput directories:")
    print(f"  Tokens:     {os.path.join(datasets.hparams.data_root, tokens_code_path)}")
    print(f"  Embeddings: {os.path.join(datasets.hparams.data_root, embeddings_code_path)}\n")

    # create model
    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae
    print("model loaded")

    # Strict load vae model
    assert cfg.TRAIN.PRETRAINED_VAE is not None
    load_pretrained_vae(cfg, model)

    if cfg.ACCELERATOR == "gpu":
        model = model.to('cuda')

    # Set model to eval mode to disable dropout
    model.eval()
    torch.set_grad_enabled(False)
    print("Model set to eval mode (dropout disabled)")

    first_save = True  # Flag for sanity check on first file

    for batch in tqdm(datasets.train_dataloader(),
                      desc=f'motion tokenize'):
        # Handle different batch formats:
        # - Original pipeline: batch['name'] is a string
        # - H2S token dataset: batch['name'] may be bool, use batch['text'] instead
        names = batch['name']
        if isinstance(names[0], bool):
            # H2S Text2MotionDatasetToken: name is in 'text' field due to collate mapping
            names = batch['text']

        # Get source dataset (for subdirectory organization)
        src = batch.get('src', ['default'])[0]

        pose = batch['motion']
        pose = pose.cuda().float()

        if pose.shape[1] == 0:
            continue

        # Encode motion to tokens
        target, _ = model.vae.encode(pose)

        # Prepare output paths
        output_dir = os.path.join(datasets.hparams.data_root, tokens_code_path, src)
        target_path = os.path.join(output_dir, names[0] + '.npy')
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)

        embeddings_output_dir = os.path.join(datasets.hparams.data_root, embeddings_code_path, src)
        embeddings_path = os.path.join(embeddings_output_dir, names[0] + '.npy')
        Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)

        # Check if R-VQVAE (3D output) or single-stage VQ (2D output)
        is_rvqvae = target.dim() == 3  # [B, T', num_quantizers]

        if is_rvqvae:
            # R-VQVAE mode: [B, T', num_quantizers]
            num_quantizers = target.shape[-1]

            # Full version: all quantizers
            target_full = target.cpu().numpy()

            # Get embeddings by summing all quantizer stage embeddings
            embeddings = get_rvqvae_embeddings(model.vae, target)

            if first_save:
                print(f"\n[R-VQVAE mode, {num_quantizers} quantizers]")
                print(f"  Full codes shape: {target_full.shape}")
                print(f"  Embeddings shape: {embeddings.shape}")

            # Save full version
            if target_full.dtype not in [np.int32, np.int64]:
                target_full = target_full.astype(np.int64)
            np.save(target_path, target_full)

        else:
            # Single-stage VQVAE mode: [B, T']
            target_np = target.cpu().numpy()

            # Get embeddings by dequantizing token IDs
            embeddings = model.vae.quantizer.dequantize(target).cpu().numpy()

            if first_save:
                print(f"\n[Single-stage VQ mode]")
                print(f"  Tokens shape:     {target_np.shape}")
                print(f"  Embeddings shape: {embeddings.shape}")

            # Save tokens
            if target_np.dtype not in [np.int32, np.int64]:
                target_np = target_np.astype(np.int64)
            np.save(target_path, target_np)

        # Save embeddings
        if embeddings.dtype not in [np.float32, np.float16]:
            embeddings = embeddings.astype(np.float32)
        np.save(embeddings_path, embeddings)

        # Sanity check on first file
        if first_save:
            print("\n" + "="*80)
            print("SANITY CHECK - First file validation:")
            print("="*80)

            loaded_tokens = np.load(target_path)
            print(f"✓ Token file saved and reloaded successfully")
            print(f"  File: {os.path.basename(target_path)}")
            print(f"  Shape: {loaded_tokens.shape}")
            print(f"  Dtype: {loaded_tokens.dtype}")
            print(f"  Min value: {loaded_tokens.min()}")
            print(f"  Max value: {loaded_tokens.max()}")
            print(f"  Sample values: {loaded_tokens.flatten()[:10]}")

            loaded_embeddings = np.load(embeddings_path)
            print(f"\n✓ Embeddings file saved and reloaded successfully")
            print(f"  File: {os.path.basename(embeddings_path)}")
            print(f"  Shape: {loaded_embeddings.shape}")
            print(f"  Dtype: {loaded_embeddings.dtype}")
            print(f"  Min/Max: {loaded_embeddings.min():.4f} / {loaded_embeddings.max():.4f}")

            print("="*80 + "\n")
            first_save = False

    print('\n' + '='*80)
    print('Motion tokenization COMPLETED!')
    print('='*80)
    print(f'Token IDs saved to: {os.path.join(datasets.hparams.data_root, tokens_code_path)}')
    print(f'Embeddings saved to: {os.path.join(datasets.hparams.data_root, embeddings_code_path)}')
    print('='*80)


if __name__ == "__main__":
    main()
