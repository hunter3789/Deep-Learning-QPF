from pathlib import Path

import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parent

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA not available, using CPU")
    device = torch.device("cpu")

# Function to compute image gradients
def compute_gradient(logits):
    grad_x = torch.abs(logits[:, 1:, :] - logits[:, :-1, :])
    grad_y = torch.abs(logits[:, :, 1:] - logits[:, :, :-1])
    return grad_x, grad_y

class RegressorLoss(nn.Module):
    #def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        MSE Loss
        Args:
            logits: tensor logits
            target: tensor targets

        Returns:
            tensor, scalar loss
        """
        loss1 = nn.L1Loss(reduction='none')
        loss2 = nn.MSELoss(reduction='none')

        #weights = weights[mask]

        # Compute gradients for the output and target
        #logits_grad_x, logits_grad_y = compute_gradient(logits)
        #target_grad_x, target_grad_y = compute_gradient(target)

        # Apply mask to gradients
        #masked_output_grad_x = logits_grad_x * mask[:, 1:, :]
        #masked_output_grad_y = logits_grad_y * mask[:, :, 1:]
        #masked_target_grad_x = target_grad_x * mask[:, 1:, :]
        #masked_target_grad_y = target_grad_y * mask[:, :, 1:]

        # Compute the gradient loss on the masked region
        #loss_x = torch.mean(torch.abs(masked_output_grad_x - masked_target_grad_x))
        #loss_y = torch.mean(torch.abs(masked_output_grad_y - masked_target_grad_y))

        output = 0.8 * loss1(logits, target) + 0.2 * loss2(logits, target)
        output = output[mask]

        #output = torch.mean(output*weights)
        output = torch.mean(output)

        #constant = 1.5
        #output += loss_x + loss_y
        return output

class CrossEntropyLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        
        #loss = nn.CrossEntropyLoss(reduction='none')
        loss = nn.CrossEntropyLoss()
        output = loss(logits, target)
        #output = output[mask].mean()

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
                #self.norm,
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
            #print(x.shape, self.skip(x).shape, self.model(x).shape)
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
            #print(x.shape, self.skip(x).shape, self.model(x).shape)
            return self.skip(x) + self.model(x)              

    def __init__(
        self,
        in_channels: int = 80,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        # TODO: implement
        #pass
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
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        # TODO: replace with actual forward pass
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
        
        #return logits, raw_depth
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
            #print(x.shape, self.skip(x).shape, self.model(x).shape)
            return self.skip(x) + self.model(x)    

    def __init__(
        self,
        in_channels: int = 81,
        num_classes: int = 2,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        # TODO: implement
        #pass
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
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        # TODO: replace with actual forward pass

        x = torch.cat((a,b),1)
        for down_layer in self.down_layers:
            x = down_layer(x)
       
        #return logits, raw_depth
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
    """
    Called by the grader to load a pre-trained model by name
    """
    #print(MODEL_DIR)
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
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) == m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = MODEL_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
