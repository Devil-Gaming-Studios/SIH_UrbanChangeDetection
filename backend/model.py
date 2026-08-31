import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        self.stage0 = ConvNormAct(in_channels, base_channels)
        self.stage1 = ConvNormAct(base_channels, base_channels * 2, stride=2)
        self.stage2 = ConvNormAct(base_channels * 2, base_channels * 4, stride=2)
        self.stage3 = ConvNormAct(base_channels * 4, base_channels * 8, stride=2)

    def forward(self, x: torch.Tensor):
        f0 = self.stage0(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f0, f1, f2, f3


class SatelliteChangeNet(nn.Module):
    """Temporal, QA-aware satellite image change segmentation network."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        red_index: int = 0,
        green_index: int = 1,
        nir_index: int = 3,
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.red_index = int(red_index)
        self.green_index = int(green_index)
        self.nir_index = int(nir_index)

        b = self.base_channels
        d = b * 8
        attention_dim = max(8, d // 8)

        self.encoder = SharedEncoder(self.in_channels, b)

        self.occlusion_gate = nn.Sequential(
            ConvNormAct(d + 1, d // 2),
            nn.Conv2d(d // 2, 1, kernel_size=1),
        )

        self.query = nn.Linear(d, attention_dim, bias=False)
        self.key = nn.Linear(d, attention_dim, bias=False)
        self.value = nn.Linear(d, d, bias=False)

        self.temporal_refiner = nn.Sequential(
            ConvNormAct(d * 2, d),
            nn.Conv2d(d, d, kernel_size=1),
        )
        self.residual_addback = nn.Sequential(
            ConvNormAct(d * 2, d),
            nn.Conv2d(d, d, kernel_size=1),
        )

        self.dec3 = ConvNormAct(d * 2, d)
        self.dec2 = ConvNormAct(d + b * 4, b * 4)
        self.dec1 = ConvNormAct(b * 4 + b * 2, b * 2)
        self.dec0 = ConvNormAct(b * 2 + b, b)

        self.segmentation_head = nn.Sequential(
            ConvNormAct(b, b),
            nn.Conv2d(b, self.num_classes, kernel_size=1),
        )

        self.spectral_projection = ConvNormAct(2, b)
        self.spectral_gate = nn.Sequential(
            ConvNormAct(b * 2, b),
            nn.Conv2d(b, b, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spectral_residual = nn.Sequential(
            ConvNormAct(b, b),
            nn.Conv2d(b, self.num_classes, kernel_size=1),
        )

    def get_config(self) -> Dict[str, int]:
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "red_index": self.red_index,
            "green_index": self.green_index,
            "nir_index": self.nir_index,
        }

    @staticmethod
    def _epoch_view(x: torch.Tensor, batch: int, epochs: int) -> torch.Tensor:
        return x.reshape(batch, epochs, *x.shape[1:])

    @staticmethod
    def _aggregate_temporal(x: torch.Tensor):
        mean = x.mean(dim=1)
        delta = (x - mean.unsqueeze(1)).abs().mean(dim=1)
        return mean, delta

    def _spectral_indices(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, _, _ = x.shape
        eps = 1e-5

        if c > max(self.red_index, self.green_index, self.nir_index):
            red = x[:, :, self.red_index]
            green = x[:, :, self.green_index]
            nir = x[:, :, self.nir_index]
            ndvi = (nir - red) / (nir + red + eps)
            ndwi = (green - nir) / (green + nir + eps)
        else:
            red = x[:, :, 0]
            green = x[:, :, min(1, c - 1)]
            blue = x[:, :, min(2, c - 1)]
            ndvi = (green - red) / (green + red + eps)
            ndwi = (blue - green) / (blue + green + eps)

        ndvi_delta = (ndvi - ndvi.mean(dim=1, keepdim=True)).abs().mean(dim=1)
        ndwi_delta = (ndwi - ndwi.mean(dim=1, keepdim=True)).abs().mean(dim=1)
        return torch.stack((ndvi_delta, ndwi_delta), dim=1).clamp(0.0, 2.0)

    def _spectral_indices_per_epoch(self, x: torch.Tensor):
        """Return per-epoch NDVI and NDWI maps (not just the temporal delta).

        Used by the API layer to render an NDVI/NDWI map for each individual
        year/epoch, in addition to the change magnitude used internally.
        Returns two tensors shaped [B, T, H, W].
        """
        b, t, c, _, _ = x.shape
        eps = 1e-5

        if c > max(self.red_index, self.green_index, self.nir_index):
            red = x[:, :, self.red_index]
            green = x[:, :, self.green_index]
            nir = x[:, :, self.nir_index]
            ndvi = (nir - red) / (nir + red + eps)
            ndwi = (green - nir) / (green + nir + eps)
        else:
            red = x[:, :, 0]
            green = x[:, :, min(1, c - 1)]
            blue = x[:, :, min(2, c - 1)]
            ndvi = (green - red) / (green + red + eps)
            ndwi = (blue - green) / (blue + green + eps)

        return ndvi, ndwi

    def forward(
        self,
        x: torch.Tensor,
        qa_mask: Optional[torch.Tensor] = None,
        return_indices: bool = False,
    ):
        """Predict a change map.

        Args:
            x: Tensor shaped [batch, epochs, channels, height, width].
            qa_mask: Optional validity tensor shaped [batch, epochs, 1, H, W].
            return_indices: if True, also returns per-epoch (ndvi, ndwi) maps
                for the API/report layer to visualize.
        """
        if x.ndim != 5:
            raise ValueError("x must have shape [B, T, C, H, W]")
        batch, epochs, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, received {channels}"
            )
        if epochs < 2:
            raise ValueError("At least two temporal epochs are required")

        flat_x = x.reshape(batch * epochs, channels, height, width)
        f0, f1, f2, f3 = self.encoder(flat_x)
        low_h, low_w = f3.shape[-2:]
        feature_dim = f3.shape[1]

        f3_t = self._epoch_view(f3, batch, epochs)
        if qa_mask is None:
            qa_mask = torch.ones(
                batch, epochs, 1, height, width, dtype=x.dtype, device=x.device,
            )
        else:
            if qa_mask.ndim == 4:
                qa_mask = qa_mask.unsqueeze(2)
            if qa_mask.shape[:3] != (batch, epochs, 1):
                raise ValueError("qa_mask must have shape [B, T, 1, H, W]")
            qa_mask = qa_mask.to(dtype=x.dtype, device=x.device)
            if qa_mask.shape[-2:] != (height, width):
                qa_mask = F.interpolate(
                    qa_mask.reshape(batch * epochs, 1, *qa_mask.shape[-2:]),
                    size=(height, width),
                    mode="nearest",
                ).reshape(batch, epochs, 1, height, width)

        qa_low = F.interpolate(
            qa_mask.reshape(batch * epochs, 1, height, width),
            size=(low_h, low_w),
            mode="nearest",
        ).reshape(batch, epochs, 1, low_h, low_w).clamp(0.0, 1.0)

        occlusion_logits = self.occlusion_gate(
            torch.cat((f3, 1.0 - qa_low.reshape(batch * epochs, 1, low_h, low_w)), dim=1)
        )
        learned_occlusion = torch.sigmoid(occlusion_logits).reshape(
            batch, epochs, 1, low_h, low_w
        )
        quality = qa_low * (1.0 - learned_occlusion)

        tokens = f3_t.mean(dim=(-2, -1))
        q = self.query(tokens)
        k = self.key(tokens)
        similarity = torch.matmul(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        valid_epoch = quality.mean(dim=(-1, -2, -3)) > 1e-4
        similarity = similarity.masked_fill(~valid_epoch.unsqueeze(1), -1e4)
        attention = torch.softmax(similarity, dim=-1)

        value = self.value(tokens).unsqueeze(-1).unsqueeze(-1)
        value = value.expand(-1, -1, -1, low_h, low_w)
        context = torch.einsum("bij,bjdhw->bidhw", attention, value)
        context = context + torch.einsum("bij,bjdhw->bidhw", attention, f3_t)

        refine_input = torch.cat((f3_t, context), dim=2).reshape(
            batch * epochs, feature_dim * 2, low_h, low_w
        )
        refined = f3_t + self.temporal_refiner(refine_input).reshape(
            batch, epochs, feature_dim, low_h, low_w
        )

        residual_input = torch.cat((refined, context), dim=2).reshape(
            batch * epochs, feature_dim * 2, low_h, low_w
        )
        residual = self.residual_addback(residual_input).reshape(
            batch, epochs, feature_dim, low_h, low_w
        )
        corrected = refined + (1.0 - quality) * residual

        bottleneck_mean, bottleneck_delta = self._aggregate_temporal(corrected)
        decoder = self.dec3(torch.cat((bottleneck_mean, bottleneck_delta), dim=1))

        skip0 = self._aggregate_temporal(self._epoch_view(f0, batch, epochs))[0]
        skip1 = self._aggregate_temporal(self._epoch_view(f1, batch, epochs))[0]
        skip2 = self._aggregate_temporal(self._epoch_view(f2, batch, epochs))[0]

        decoder = F.interpolate(decoder, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        decoder = self.dec2(torch.cat((decoder, skip2), dim=1))
        decoder = F.interpolate(decoder, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        decoder = self.dec1(torch.cat((decoder, skip1), dim=1))
        decoder = F.interpolate(decoder, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        decoder = self.dec0(torch.cat((decoder, skip0), dim=1))
        decoder = F.interpolate(decoder, size=(height, width), mode="bilinear", align_corners=False)

        main_logits = self.segmentation_head(decoder)

        spectral_delta = self._spectral_indices(x)
        spectral_features = self.spectral_projection(spectral_delta)
        gate = self.spectral_gate(torch.cat((decoder, spectral_features), dim=1))
        spectral_logits = self.spectral_residual(spectral_features * gate)

        logits = main_logits + 0.25 * spectral_logits

        if return_indices:
            ndvi_per_epoch, ndwi_per_epoch = self._spectral_indices_per_epoch(x)
            ndvi_delta = spectral_delta[:, 0]
            ndwi_delta = spectral_delta[:, 1]
            return logits, ndvi_delta, ndwi_delta, ndvi_per_epoch, ndwi_per_epoch

        return logits


if __name__ == "__main__":
    model = SatelliteChangeNet(in_channels=3)
    dummy_x = torch.rand(2, 2, 3, 128, 128)
    dummy_qa = torch.ones(2, 2, 1, 128, 128)
    y = model(dummy_x, dummy_qa)
    print({"input": tuple(dummy_x.shape), "output": tuple(y.shape),
           "parameters": sum(p.numel() for p in model.parameters())})
