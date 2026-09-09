import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplane import TriplaneTransformer
from .pvcnn.encoder_pc import TriPlanePC2Encoder


class SimpleTriplaneModel(nn.Module):
    """
    Minimal forward-only model that encodes a point cloud with PVCNN and
    transforms it into triplane features with a TriplaneTransformer.

    The forward output is the triplane tensor returned immediately after
    the transformer stage. Assumes PVCNN is always used.
    """

    def __init__(self, cfg, apply_tanh: bool = True) -> None:
        super().__init__()

        # Store a subset of config for potential downstream uses
        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_channels_low = cfg.triplane_channels_low
        self._verbose = False
        self._forward_hook_handles = []
        self.apply_tanh = apply_tanh

        # Build TriplaneTransformer (mirrors original settings)
        # Ensure the low-res matches the high-res after two stride-2 pools (high_res / 4)
        high_res = int(getattr(cfg.pvcnn, 'z_triplane_resolution', 128))

        triplane_low_res = getattr(cfg, 'triplane_low_res', None)
        if triplane_low_res is None:
            low_res = max(1, high_res // 4)
        else:
            low_res = triplane_low_res

        self.triplane_transformer = TriplaneTransformer(
            input_dim=cfg.pvcnn.z_triplane_channels,  # typically 256
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=low_res,
            triplane_high_res=high_res,  # typically 128
            triplane_dim=cfg.triplane_channels_high,  # e.g., 512
        )

        # Always use PVCNN for feature extraction
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.pvcnn = TriPlanePC2Encoder(
            cfg.pvcnn,
            device=device_str,
            shape_min=-1,
            shape_length=2,
            use_2d_feat=False,
        )

    def forward(self, point_cloud_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_cloud_xyz: Tensor of shape [B, N, 3]

        Returns:
            planes: Triplane tensor after transformer, shape [B, 3, C, H, W]
        """
        # PVCNN expects two inputs in the original code; both were the same tensor
        pc_feat = self.pvcnn(point_cloud_xyz, point_cloud_xyz)
        if self._verbose:
            self._print_tensor_info("pc_feat", pc_feat)

        # Transform to output triplanes and return immediately
        planes = self.triplane_transformer(pc_feat)
        if self._verbose:
            self._print_tensor_info("planes", planes)
            
        # Apply tanh activation if enabled
        if self.apply_tanh:
            planes = torch.tanh(planes)
        return planes

    def sample_triplane_features_from_points(self, feature_triplane: torch.Tensor, normalized_pos: torch.Tensor) -> torch.Tensor:
        tri_plane = torch.unbind(feature_triplane, dim=1)  # 3 tensors of shape [B, C, H, W]
        x_y = torch.cat([normalized_pos[:, :, 0:1], normalized_pos[:, :, 1:2]], dim=-1).unsqueeze(1)
        y_z = torch.cat([normalized_pos[:, :, 1:2], normalized_pos[:, :, 2:3]], dim=-1).unsqueeze(1)
        x_z = torch.cat([normalized_pos[:, :, 0:1], normalized_pos[:, :, 2:3]], dim=-1).unsqueeze(1)
        x_feat = F.grid_sample(tri_plane[0], x_y, padding_mode='border', align_corners=True)
        y_feat = F.grid_sample(tri_plane[1], y_z, padding_mode='border', align_corners=True)
        z_feat = F.grid_sample(tri_plane[2], x_z, padding_mode='border', align_corners=True)
        final_feat = (x_feat + y_feat + z_feat)
        final_feat = final_feat.squeeze(dim=2).permute(0, 2, 1)  # [B, N, C]
        return final_feat

    # -------- Verbose utilities --------
    def enable_verbose_forward(self) -> None:
        """Enable verbose printing: module execution order and output shapes."""
        if self._verbose:
            return
        self._verbose = True
        # Install hooks on submodules of pvcnn and triplane_transformer to avoid duplicating root prints
        self._register_hooks_for_branch(self.pvcnn, prefix="pvcnn")
        self._register_hooks_for_branch(self.triplane_transformer, prefix="triplane_transformer")

    def disable_verbose_forward(self) -> None:
        for handle in self._forward_hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._forward_hook_handles = []
        self._verbose = False

    def print_architecture(self) -> None:
        """Print a concise architecture summary and parameter counts."""
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters: total={total_params:,} trainable={trainable_params:,} frozen={total_params-trainable_params:,}")

    # -------- Internal helpers --------
    def _register_hooks_for_branch(self, module: nn.Module, prefix: str) -> None:
        for name, submodule in module.named_modules():
            full_name = f"{prefix}.{name}" if name else prefix
            # Skip containers that are the top of this branch (we still want their children)
            def _make_hook(tag):
                def _hook(mod, inputs, outputs):
                    self._print_module_output(tag, outputs)
                return _hook
            handle = submodule.register_forward_hook(_make_hook(full_name))
            self._forward_hook_handles.append(handle)

    def _print_tensor_info(self, name: str, tensor: torch.Tensor) -> None:
        try:
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)
            device = str(tensor.device)
            print(f"[INTERMEDIATE] {name}: shape={shape} dtype={dtype} device={device}")
        except Exception:
            print(f"[INTERMEDIATE] {name}: <unavailable>")

    def _print_module_output(self, module_name, outputs) -> None:
        def fmt(x):
            if isinstance(x, torch.Tensor):
                return f"Tensor{tuple(x.shape)} {str(x.dtype)}"
            return type(x).__name__

        if isinstance(outputs, torch.Tensor):
            desc = fmt(outputs)
        elif isinstance(outputs, (list, tuple)):
            desc = '[' + ', '.join(fmt(o) for o in outputs) + ']'
        elif isinstance(outputs, dict):
            parts = []
            for k, v in outputs.items():
                parts.append(f"{k}: {fmt(v)}")
            desc = '{' + ', '.join(parts) + '}'
        else:
            desc = fmt(outputs)
        print(f"[MODULE] {module_name} -> {desc}")
