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
        embeddings: numpy array [B, T', code_dim]
    """
    B, T_prime, num_quantizers = code_idx.shape

    embeddings = None
    for i in range(num_quantizers):
        indices = code_idx[:, :, i].reshape(-1)  # [B*T']
        quantizer = vae.quantizer._get_quantizer(i)
        z_q = quantizer.dequantize(indices)  # [B*T', code_dim]
        z_q = z_q.view(B, T_prime, vae.code_dim)

        if embeddings is None:
            embeddings = z_q
        else:
            embeddings = embeddings + z_q

    return embeddings.detach().cpu().numpy()


def get_single_vq_embeddings(vae, code_idx):
    """
    Get embeddings from single-stage VQ token indices.

    Args:
        vae: VQVae model
        code_idx: Token indices [B, T']

    Returns:
        embeddings: numpy array [B, T', code_dim]
    """
    return vae.quantizer.dequantize(code_idx).detach().cpu().numpy()


def encode_and_extract(vae, pose, is_rvqvae):
    """
    Encode pose and extract both token IDs and embeddings.

    Returns:
        tokens: numpy array, embeddings: numpy array
    """
    tokens, _ = vae.encode(pose)
    if is_rvqvae:
        embeddings = get_rvqvae_embeddings(vae, tokens)
    else:
        embeddings = get_single_vq_embeddings(vae, tokens)
    return tokens.cpu().numpy(), embeddings


def main():
    cfg = parse_args(phase="test")
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    pl.seed_everything(cfg.SEED_VALUE)

    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")

    # Determine output paths
    tokens_code_path = cfg.DATASET.CODE_PATH
    if 'TOKENS' in tokens_code_path.upper():
        embeddings_code_path = tokens_code_path.replace('TOKENS', 'EMBEDDINGS').replace('tokens', 'embeddings')
    else:
        tokens_code_path = os.path.join(tokens_code_path, 'TOKENS')
        embeddings_code_path = os.path.join(cfg.DATASET.CODE_PATH, 'EMBEDDINGS')

    print(f"\nOutput directories:")
    print(f"  Tokens:     {os.path.join(datasets.hparams.data_root, tokens_code_path)}")
    print(f"  Embeddings: {os.path.join(datasets.hparams.data_root, embeddings_code_path)}\n")

    # Create model and load pretrained VAE
    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae
    print("model loaded")

    assert cfg.TRAIN.PRETRAINED_VAE is not None
    load_pretrained_vae(cfg, model)

    if cfg.ACCELERATOR == "gpu":
        model = model.to('cuda')

    model.eval()
    torch.set_grad_enabled(False)
    print("Model set to eval mode")

    # Detect VAE mode
    has_3part = hasattr(model, 'hand_vae') and hasattr(model, 'rhand_vae')
    if has_3part:
        print("[3-part VAE mode: body + lhand + rhand]")
    else:
        print("[Single VAE mode]")

    first_save = True

    for batch in tqdm(datasets.train_dataloader(), desc='motion tokenize'):
        names = batch['name']
        if isinstance(names[0], bool):
            names = batch['text']

        src = batch.get('src', ['default'])[0]
        pose = batch['motion'].cuda().float()

        if pose.shape[1] == 0:
            continue

        # Prepare output paths
        output_dir = os.path.join(datasets.hparams.data_root, tokens_code_path, src)
        target_path = os.path.join(output_dir, names[0] + '.npy')
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)

        embeddings_output_dir = os.path.join(datasets.hparams.data_root, embeddings_code_path, src)
        embeddings_path = os.path.join(embeddings_output_dir, names[0] + '.npy')
        Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)

        if has_3part:
            # 3-part mode: split pose into body/lhand/rhand
            pose_lhand = pose[..., 30:75]
            pose_rhand = pose[..., 75:120]
            pose_body = torch.cat([pose[..., :30], pose[..., 120:]], dim=-1)

            # Encode each part
            tokens_body, _ = model.vae.encode(pose_body)
            tokens_lhand, _ = model.hand_vae.encode(pose_lhand)
            tokens_rhand, _ = model.rhand_vae.encode(pose_rhand)

            is_rvqvae = tokens_body.dim() == 3  # [B, T', num_quantizers]

            if is_rvqvae:
                # R-VQVAE: stack [B, T', num_quantizers, 3]
                target = np.stack([
                    tokens_body.cpu().numpy(),
                    tokens_lhand.cpu().numpy(),
                    tokens_rhand.cpu().numpy()
                ], axis=-1)

                # Also save Q0-only: [B, T', 3]
                target_q0 = np.stack([
                    tokens_body[:, :, 0].cpu().numpy(),
                    tokens_lhand[:, :, 0].cpu().numpy(),
                    tokens_rhand[:, :, 0].cpu().numpy()
                ], axis=-1)

                # Get embeddings: sum all quantizer stages
                emb_body = get_rvqvae_embeddings(model.vae, tokens_body)
                emb_lhand = get_rvqvae_embeddings(model.hand_vae, tokens_lhand)
                emb_rhand = get_rvqvae_embeddings(model.rhand_vae, tokens_rhand)

                # Save Q0-only version
                target_path_q0 = target_path.replace('.npy', '_q0.npy')
                np.save(target_path_q0, target_q0.astype(np.int64))
            else:
                # Single-stage: [B, T', 3]
                target = np.stack([
                    tokens_body.cpu().numpy(),
                    tokens_lhand.cpu().numpy(),
                    tokens_rhand.cpu().numpy()
                ], axis=-1)

                emb_body = get_single_vq_embeddings(model.vae, tokens_body)
                emb_lhand = get_single_vq_embeddings(model.hand_vae, tokens_lhand)
                emb_rhand = get_single_vq_embeddings(model.rhand_vae, tokens_rhand)

            # Stack embeddings: [B, T', 3, code_dim]
            embeddings = np.stack([emb_body, emb_lhand, emb_rhand], axis=-2)

        else:
            # Single VAE mode
            tokens, _ = model.vae.encode(pose)
            is_rvqvae = tokens.dim() == 3

            if is_rvqvae:
                target = tokens.cpu().numpy()
                embeddings = get_rvqvae_embeddings(model.vae, tokens)
            else:
                target = tokens.cpu().numpy()
                embeddings = get_single_vq_embeddings(model.vae, tokens)

        # Save tokens
        if target.dtype not in [np.int32, np.int64]:
            target = target.astype(np.int64)
        np.save(target_path, target)

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
            print(f"  Token file:  {os.path.basename(target_path)}")
            print(f"  Shape: {loaded_tokens.shape}, Dtype: {loaded_tokens.dtype}")
            print(f"  Min: {loaded_tokens.min()}, Max: {loaded_tokens.max()}")
            print(f"  Sample: {loaded_tokens.flatten()[:10]}")

            loaded_embeddings = np.load(embeddings_path)
            print(f"\n  Embeddings:  {os.path.basename(embeddings_path)}")
            print(f"  Shape: {loaded_embeddings.shape}, Dtype: {loaded_embeddings.dtype}")
            print(f"  Min: {loaded_embeddings.min():.4f}, Max: {loaded_embeddings.max():.4f}")

            print("="*80 + "\n")
            first_save = False

    print('\n' + '='*80)
    print('Motion tokenization COMPLETED!')
    print(f'Token IDs:  {os.path.join(datasets.hparams.data_root, tokens_code_path)}')
    print(f'Embeddings: {os.path.join(datasets.hparams.data_root, embeddings_code_path)}')
    print('='*80)


if __name__ == "__main__":
    main()
