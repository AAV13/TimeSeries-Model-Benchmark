#
# Copyright 2024 Google LLC
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class Patches(nn.Module):
    """Create patches from time series."""

    def __init__(self, context_len: int, patch_len: int):
        super().__init__()
        self.context_len = context_len
        self.patch_len = patch_len
        if self.context_len % self.patch_len != 0:
            raise ValueError("context_len must be divisible by patch_len")
        self.num_patches = self.context_len // self.patch_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create patches from time series.

        Args:
            x (torch.Tensor): Time series tensor of shape (B, L, D).

        Returns:
            torch.Tensor: Patches tensor of shape (B, N, P, D).
        """
        x = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        return x


class Unpatches(nn.Module):
    """Unpatch time series."""

    def __init__(self, num_patches: int, patch_len: int):
        super().__init__()
        self.num_patches = num_patches
        self.patch_len = patch_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unpatch time series.

        Args:
            x (torch.Tensor): Patches tensor of shape (B, N, P, D).

        Returns:
            torch.Tensor: Time series tensor of shape (B, L, D).
        """
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


class Mask(nn.Module):
    """Masking for time series."""

    def __init__(self, mask_ratio: float, patch_len: int, mask_type: str = "random"):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_len = patch_len
        self.mask_type = mask_type

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Masking for time series.

        Args:
            x (torch.Tensor): Patches tensor of shape (B, N, P, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Masked patches tensor and mask.
        """
        if self.mask_type == "random":
            return self.random_mask(x)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

    def random_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Random masking for time series.

        Args:
            x (torch.Tensor): Patches tensor of shape (B, N, P, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Masked patches tensor and mask.
        """
        b, n, p, d = x.shape
        num_masked = int(self.mask_ratio * n)
        if num_masked == 0:
            return x, torch.zeros((b, n), device=x.device, dtype=torch.bool)
        masked_indices = (
            torch.rand(b, n, device=x.device).argsort(dim=-1)[:, :num_masked]
        )
        mask = torch.zeros((b, n), device=x.device, dtype=torch.bool)
        mask.scatter_(dim=1, index=masked_indices, value=True)
        x_masked = torch.where(
            mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, p, d),
            torch.zeros_like(x),
            x,
        )
        return x_masked, mask


class PatchEmbedding(nn.Module):
    """Patch embedding for time series."""

    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embedding for time series.

        Args:
            x (torch.Tensor): Patches tensor of shape (B, N, D, P).

        Returns:
            torch.Tensor: Embedded patches tensor of shape (B, N, D_MODEL).
        """
        # The projection correctly operates on the last dimension (P)
        x = self.projection(x) # Output shape is (B, N, D, d_model)

        # THE FIX: Squeeze the redundant D dimension (dim=2) to make it a 3D tensor
        x = x.squeeze(2) # Output shape is (B, N, d_model)
        
        return x

class PositionalEmbedding(nn.Module):
    """Positional embedding for time series."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Positional embedding for time series.

        Args:
            x (torch.Tensor): Embedded patches tensor of shape (B, N, D_MODEL).

        Returns:
            torch.Tensor: Embedded patches tensor with positional encoding.
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for time series."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer encoder for time series.

        Args:
            x (torch.Tensor): Embedded patches tensor of shape (B, N, D_MODEL).

        Returns:
            torch.Tensor: Encoded tensor of shape (B, N, D_MODEL).
        """
        x = self.encoder(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder for time series."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Transformer decoder for time series.

        Args:
            tgt (torch.Tensor): Target tensor of shape (B, N, D_MODEL).
            memory (torch.Tensor): Memory tensor of shape (B, N, D_MODEL).

        Returns:
            torch.Tensor: Decoded tensor of shape (B, N, D_MODEL).
        """
        x = self.decoder(tgt, memory)
        return x


class LinearProjection(nn.Module):
    """Linear projection for time series."""

    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.projection = nn.Linear(d_model, patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear projection for time series.

        Args:
            x (torch.Tensor): Encoded tensor of shape (B, N, D_MODEL).

        Returns:
            torch.Tensor: Projected tensor of shape (B, N, P).
        """
        x = self.projection(x)
        return x.unsqueeze(-1)


class TimesFM(nn.Module):
    def __init__(
        self,
        context_len: int,
        horizon_len: int,
        input_patch_len: int,
        output_patch_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        backend: str,
        freq_token_vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.backend = backend
        self.freq_token_vocab_size = freq_token_vocab_size

        self.input_num_patches = self.context_len // self.input_patch_len
        self.output_num_patches = self.horizon_len // self.output_patch_len

        self.input_patcher = Patches(self.context_len, self.input_patch_len)
        self.output_patcher = Patches(self.horizon_len, self.output_patch_len)

        self.input_unpatcher = Unpatches(
            self.input_num_patches, self.input_patch_len
        )
        self.output_unpatcher = Unpatches(
            self.output_num_patches, self.output_patch_len
        )
        self.input_mask = Mask(
            mask_ratio=0.5, patch_len=self.input_patch_len, mask_type="random"
        )
        self.input_embedding = PatchEmbedding(self.input_patch_len, self.d_model)
        self.positional_embedding = PositionalEmbedding(self.d_model)
        if self.freq_token_vocab_size is not None:
            self.freq_embedding = nn.Embedding(
                self.freq_token_vocab_size, self.d_model
            )

        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.decoder = TransformerDecoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        self.output_projection = LinearProjection(self.d_model, self.output_patch_len)

    def forward(self, x_in, x_out, f_in: Optional[torch.Tensor] = None):
        x_in = self.input_patcher(x_in.unsqueeze(-1))
        x_in, mask = self.input_mask(x_in)
        x_in = self.input_embedding(x_in)
        x_in = self.positional_embedding(x_in)
        if self.freq_token_vocab_size is not None:
            f_in_embedded = self.freq_embedding(f_in)
            x_in = x_in + f_in_embedded.unsqueeze(1).repeat(1, x_in.shape[1], 1)

        enc_out = self.encoder(x_in)
        dec_in = torch.zeros(
            (x_out.shape[0], self.output_num_patches, self.d_model)
        ).to(x_out.device)
        dec_in = self.positional_embedding(dec_in)

        if self.freq_token_vocab_size is not None:
            f_in_embedded = self.freq_embedding(f_in)
            dec_in = dec_in + f_in_embedded.unsqueeze(1).repeat(1, dec_in.shape[1], 1)

        dec_out = self.decoder(dec_in, enc_out)
        y = self.output_projection(dec_out)
        y = self.output_unpatcher(y).squeeze(-1)
        return y, mask

    def forecast(self, x_in, f_in: Optional[torch.Tensor] = None):
        x_in_patched = self.input_patcher(x_in.unsqueeze(-1))
        x_in_embedded = self.input_embedding(x_in_patched)
        x_in_embedded = self.positional_embedding(x_in_embedded)

        if self.freq_token_vocab_size is not None:
            f_in_embedded = self.freq_embedding(f_in)
            x_in_embedded = x_in_embedded + f_in_embedded.unsqueeze(1).repeat(
                1, x_in_embedded.shape[1], 1
            )
        enc_out = self.encoder(x_in_embedded)

        dec_in = torch.zeros(
            (x_in.shape[0], self.output_num_patches, self.d_model)
        ).to(x_in.device)
        dec_in = self.positional_embedding(dec_in)

        if self.freq_token_vocab_size is not None:
            f_in_embedded = self.freq_embedding(f_in)
            dec_in = dec_in + f_in_embedded.unsqueeze(1).repeat(1, dec_in.shape[1], 1)

        dec_out = self.decoder(dec_in, enc_out)
        y = self.output_projection(dec_out)
        y = self.output_unpatcher(y).squeeze(-1)
        return y, None
    
    def load_from_checkpoint(self, path):
        chkpt = torch.load(path, map_location=torch.device('cpu'))
        # Adjust keys
        new_chkpt = {}
        for k, v in chkpt.items():
            if 'model.' in k:
                new_chkpt[k.replace('model.', '')] = v
            else:
                new_chkpt[k] = v
        self.load_state_dict(new_chkpt, strict=False)

    @classmethod
    def from_pretrained(cls, checkpoint_path, device):
        # These are the parameters for the 200M model.
        model = cls(
            context_len=512,
            horizon_len=256,
            input_patch_len=32,
            output_patch_len=128,
            d_model=1280,
            num_layers=20,
            num_heads=16,
            d_ff=1280,
            dropout=0.1,
            backend=device,
        )
        model.load_from_checkpoint(checkpoint_path)
        return model