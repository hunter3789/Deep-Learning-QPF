from pathlib import Path

import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parent

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA not available, using CPU")
    device = torch.device("cpu")

class RegressorLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: tensor logits
            target: tensor targets

        Returns:
            tensor, scalar loss
        """
        loss1 = nn.L1Loss(reduction='none')
        loss2 = nn.MSELoss(reduction='none')

        output = 0.8 * loss1(logits, target) + 0.2 * loss2(logits, target)
        output = output[mask]

        output = torch.mean(output)

        return output

class CrossEntropyLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        
        loss = nn.CrossEntropyLoss()
        output = loss(logits, target)

        return output

class Regressor(torch.nn.Module): 
    class BlockDown(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1)//2
            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)    
            self.norm = torch.nn.BatchNorm2d(out_channels)            
            self.relu = torch.nn.ReLU()

            self.model = torch.nn.Sequential(
                self.c1,
                self.relu,
                self.c2,
                self.norm,
                self.relu,
            )

            if stride != 1 or in_channels != out_channels:
                self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):
            return self.skip(x) + self.model(x)    

    class BlockUp(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1)//2
            self.c1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)            
            self.norm = torch.nn.BatchNorm2d(out_channels)            
            self.relu = torch.nn.ReLU()

            self.model = torch.nn.Sequential(
                self.c1,
                self.norm,
                self.relu,
            )

            if stride != 1 or in_channels != out_channels:
                self.skip = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x):        
            return self.skip(x) + self.model(x)              

    def __init__(
        self,
        in_channels: int = 80,
    ):
        super().__init__()

        up_layers = []
        down_layers = []
        skip_layers = []
        n_blocks = 2
        out_channels = 128

        down_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=2, padding=5, bias=False))

        c1 = out_channels
        for _ in range(n_blocks):
            c2 = c1 * 2
            down_layers.append(self.BlockDown(c1, c2, stride=2))
            c1 = c2

        for _ in range(n_blocks):        
            c2 = int(c1 / 2)
            up_layers.append(self.BlockUp(c1, c2, stride=2))
            skip_layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0),
                torch.nn.BatchNorm2d(c2), 
                torch.nn.ReLU()                
            ))
            c1 = c2
        
        up_layers.append(torch.nn.ConvTranspose2d(c1, in_channels, kernel_size=11, stride=2, padding=5, output_padding=1))

        self.up_layers = torch.nn.ModuleList(up_layers)
        self.down_layers = torch.nn.ModuleList(down_layers)
        self.skip_layers = torch.nn.ModuleList(skip_layers)
  
        self.regressor = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        layers = []
        for down_layer in self.down_layers:
            x = down_layer(x)
            layers.append(x)

        for i in range(len(layers)):
            y = layers[len(layers) - i - 1]
            if i > 0:     
                y = torch.cat([x, y], dim=1)   
                y = self.skip_layers[i-1](y)

            x = self.up_layers[i](y)
        
        return self.regressor(x)      

class Discriminator(torch.nn.Module): 
    class BlockDown(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1)//2
            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.norm = torch.nn.BatchNorm2d(out_channels)            
            self.relu = torch.nn.ReLU()

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
        num_classes: int = 2,
    ):
        super().__init__()

        up_layers = []
        down_layers = []
        skip_layers = []
        n_blocks = 5
        out_channels = 128

        down_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=2, padding=5, bias=False))

        c1 = out_channels
        for _ in range(n_blocks):
            if c1 < 512:
                c2 = c1 * 2
            down_layers.append(self.BlockDown(c1, c2, stride=2))
            c1 = c2

        self.down_layers = torch.nn.ModuleList(down_layers)
  
        self.patch = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(c2, num_classes, kernel_size=1, stride=1, padding=0),
        )

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


def save_model(model: torch.nn.Module) -> str:
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) == m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = MODEL_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path

