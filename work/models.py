from pathlib import Path

import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parent

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA not available, using CPU")
    device = torch.device("cpu")

class MultiHeadLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCEloss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred: torch.Tensor, logits: torch.Tensor, logits_lgt: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, detected: torch.Tensor, lgt: torch.Tensor, weights: torch.Tensor, lambda_occ: float = 1.0, lambda_amt: float = 1.0, lambda_lgt: float = 1.0) -> torch.Tensor:
        """
        Combined loss used for the multi-head model:
        - Occurrence (binary rain detection)
        - Amount (regression for rainfall)
        - Lightning detection
        Includes pixel-wise mask and custom pixel-wise weights.
        """
        weights = weights[mask]

        # Occurrence (binary rain detection)
        logits_loss = self.BCEloss(logits, detected)
        logits_loss = logits_loss[mask]
        logits_loss = torch.mean(logits_loss*weights)

        # Lightning detection
        lgt_loss = self.BCEloss(logits_lgt, lgt)
        lgt_loss = lgt_loss[mask]
        lgt_loss = torch.mean(lgt_loss*weights)

        # Amount (regression for rainfall)
        all_pixel_loss = self.mse_loss(pred, target)

        pred_loss = all_pixel_loss[mask]
        pred_loss = torch.mean(pred_loss*weights)

        return lambda_occ * logits_loss + lambda_amt * pred_loss + lambda_lgt * lgt_loss

class BCELoss(nn.Module):
    """Simple BCE loss wrapper for binary classification."""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:     
        output = self.loss(logits, target)

        return output

class Regressor(torch.nn.Module):
    """
    U-Net-like multi-head model for:
        - Rainfall amount regression
        - Rain detection (occurrence)
        - Lightning detection
    """
    class BlockDown(nn.Module):
        def __init__(self, in_channels, out_channels, stride, dropout_prob=0.0):
            super().__init__()
            k = 3
            p = (k-1)//2
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, k, stride, p),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity(),

                nn.Conv2d(out_channels, out_channels, k, 1, p),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride) if (stride != 1 or in_channels != out_channels) else nn.Identity()

        def forward(self, x):
            return self.skip(x) + self.model(x)

    class BlockUp(nn.Module):
        def __init__(self, in_channels, out_channels, stride, dropout_prob=0.0, mode="bilinear"):
            super().__init__()
            self.mode = mode
            k = 3
            p = (k-1)//2
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

            self.proj_up = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        def forward(self, x, skip):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode=self.mode,
                              align_corners=False if self.mode == "bilinear" else None)

            x = self.proj_up(x)            
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=80, base_channels=128, n_blocks=2, dropout_prob=0.0, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.skip_channels = []
        max_channels = 512

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(in_channels, base_channels, kernel_size=11, stride=2, padding=5))

        c1 = base_channels
        for i in range(n_blocks):
            c2 = min(c1 * 2, max_channels)
            # Add dropout only in deeper layers (i > 0 to skip first block)
            self.encoder.append(self.BlockDown(c1, c2, stride=2, dropout_prob=dropout_prob if i > 0 else 0.0))
            self.skip_channels.append(c1)
            c1 = c2

        # Decoder
        self.decoder = nn.ModuleList()
        # Then reverse for decoder
        for i, c_skip in enumerate(reversed(self.skip_channels)):
            self.decoder.append(self.BlockUp(c1, c_skip, stride=2, dropout_prob=dropout_prob if i < 1 else 0.0, mode=self.mode))
            c1 = c_skip

        # Final upsampling (no skip connection here)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1, in_channels, kernel_size=3, padding=1, bias=False),
        )

        # Regressor head
        self.regressor = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Rain Detection head
        self.detector = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Lightning Detection head
        self.lgt = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        out = x

        # Encoder
        for layer in self.encoder:
            out = layer(out)
            skips.append(out)

        # Remove deepest feature (bottleneck) from skip list
        skips = skips[:-1][::-1]

        # Decoder
        for i, layer in enumerate(self.decoder):
            out = layer(out, skips[i])

        # Final upsample (no skip connection)
        out = self.final_up(out)

        # Final regression
        return self.regressor(out), self.detector(out), self.lgt(out)

class Discriminator(torch.nn.Module): 
    """
    PatchGAN-style discriminator for GAN training.
    Takes (input, prediction) concatenated as channels.
    """
    class BlockDown(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1)//2
            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.norm = torch.nn.BatchNorm2d(out_channels)            
            self.relu = torch.nn.LeakyReLU(0.2, inplace=True)

            self.model = torch.nn.Sequential(
                self.c1,
                self.norm,
                self.relu,
            )

            if stride != 1 or in_channels != out_channels:
                self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            return self.skip(x) + self.model(x)    

    def __init__(
        self,
        in_channels: int = 81,
        num_classes: int = 1,
    ):
        super().__init__()

        up_layers = []
        down_layers = []
        skip_layers = []
        n_blocks = 3
        out_channels = 128

        down_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=2, padding=5, bias=False))

        c1 = out_channels
        for _ in range(n_blocks):
            if c1 < 512:
                c2 = c1 * 2
            down_layers.append(self.BlockDown(c1, c2, stride=2))
            c1 = c2

        self.down_layers = torch.nn.ModuleList(down_layers)
  
        self.patch = torch.nn.Conv2d(c2, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        x = torch.cat((a,b),1)
        for down_layer in self.down_layers:
            x = down_layer(x)
       
        return self.patch(x)      

MODEL_FACTORY = {
    "regressor": Regressor,
    "discriminator": Discriminator,
}

def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = MODEL_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    return m

def load_optimizer(model_name: str, optimizer) -> torch.nn.Module:
    model_path = MODEL_DIR / f"{model_name}_optim.th"
    optimizer.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))

    return optimizer

def save_model(model: torch.nn.Module, optimizer) -> str:
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) == m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = MODEL_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    output_path2 = MODEL_DIR / f"{model_name}_optim.th"
    torch.save(optimizer.state_dict(), output_path2)

    return output_path

if __name__ == "__main__":
    debug_model()
