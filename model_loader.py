import torch

from TROLOLO.TROLOLO_pyramid import TROLOLO

def load_model(model_path=None, device='cuda',compilation=True):
    torch._dynamo.config.disable = not compilation

    quantize = False
    """trololo = TROLOLO(seq_length=64,
                      num_layers=8,
                      num_heads=48,
                      embed_dim=192,
                      mlp_dim=512,
                      n_class_tokens=2,
                      num_classes=1,
                      mlp_rank=0.1,
                      qkv_rank=0.2,
                      attnproj_rank=0.1,
                      sequence_pyramid=[],
                      attn_rank_pyramid=[],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=1.0,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits=None if not quantize else 8
                      )
    """
    trololo = TROLOLO(seq_length=64,
                      num_layers=8,
                      num_heads=48,
                      embed_dim=192,
                      mlp_dim=192,
                      n_class_tokens=2,
                      num_classes=1,
                      mlp_rank=0.1,
                      qkv_rank=0.2,
                      attnproj_rank=0.1,
                      sequence_pyramid=[],
                      attn_rank_pyramid=[],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=0.8,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits=None if not quantize else 8
                      )
    # Load checkpoint
    if model_path is not None:
        """Load trained TROLOLO model"""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        trololo.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    trololo.to(device)
    return trololo

