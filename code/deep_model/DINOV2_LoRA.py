import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_model.LoRA import LoRA
import math

class DINOV2LoRA(nn.Module):
    def __init__(self, output_dim, vit_type, low_rank, use_lora=True, freeze_vit=False):
        super().__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_' + vit_type + '14')
        self.use_lora = use_lora
       
        if freeze_vit:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False  # Freeze all parameters

        # Set the hidden dimension based on the ViT variant
        hidden_dim = {
            'vits': 384,
            'vitb': 768,
            'vitl': 1024,
            'vitg': 1536
        }[vit_type]

        if use_lora:
            self.lora_layers = list(range(len(self.encoder.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.encoder.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, low_rank)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, low_rank)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

        # Dropout based on ViT variant
        self.dropout = nn.Dropout(p=0.1 if vit_type in ['vits', 'vitb'] else 0.15 if vit_type == 'vitl' else 0.2)
        self.clf = nn.Linear(hidden_dim, output_dim)
        
    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def save_parameters(self, filename: str) -> None:
        """Save the LoRA weights and decoder weights to a .pt file

        Args:
            filename (str): Filename of the weights
        """
        w_a, w_b = {}, {}
        if self.use_lora:
            w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
            w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}
        torch.save({**w_a, **w_b}, filename)


    def load_parameters(self, filename: str) -> None:
        """Load the LoRA and decoder weights from a file

        Args:
            filename (str): File name of the weights
        """
        state_dict = torch.load(filename)

        # Load the LoRA parameters
        if self.use_lora:
            for i, w_A_linear in enumerate(self.w_a):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_b):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_B_linear.weight = nn.Parameter(saved_tensor)

       

    def forward(self, x):
        x = self.encoder(x)
        # print(x.size())
        # x = torch.mean(x, dim=-1)
        # x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # x = torch.squeeze(x)
        x = self.dropout(x)
        x = self.clf(x)
        return x

