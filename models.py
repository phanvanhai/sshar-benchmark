"""
models.py

Model definitions for WiFi HAR Benchmark
Current version:
    Skeleton only.
Each model will be implemented later.
Author:
"""

import torch
import torch.nn as nn

# ============================================================
# Model List
# ============================================================
SUPPORTED_MODELS = [
    "mlp",
    "cnn5",
    "bilstm",
    "vit",
    "resnet18",
    "cnn_gru",
    "bimamba",
]


def list_models():
    """
    Return supported model names.
    """
    return SUPPORTED_MODELS.copy()


# ============================================================
# Base Class
# ============================================================
class BaseModel(nn.Module):
    """
    Base class of all benchmark models.
    """
    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):
    """
    Simple MLP baseline.

    Input
    -----
    (batch, C, T)

    Output
    ------
    (batch, num_classes)
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        hidden_dim=512,
        dropout=0.5,
    ):
        super().__init__()

        input_dim = input_shape[0] * input_shape[1]

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim // 2,
                num_classes,
            ),
        )

    def forward(self, x):

        return self.classifier(x)


# ============================================================
# CNN-5
# ============================================================

class CNN5(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
        )
        raise NotImplementedError(
            "CNN5 has not been implemented yet."
        )

    def forward(self, x):
        raise NotImplementedError


# ============================================================
# BiLSTM
# ============================================================
class BiLSTM(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
        )

        raise NotImplementedError(
            "BiLSTM has not been implemented yet."
        )

    def forward(self, x):
        raise NotImplementedError


# ============================================================
# Vision Transformer
# ============================================================
class ViT(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
        )

        raise NotImplementedError(
            "ViT has not been implemented yet."
        )

    def forward(self, x):
        raise NotImplementedError


# ============================================================
# ResNet18
# ============================================================
class ResNet18(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
        )

        raise NotImplementedError(
            "ResNet18 has not been implemented yet."
        )

    def forward(self, x):
        raise NotImplementedError

# ============================================================
# CNN + GRU
# ============================================================
class CNN_GRU(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
        )

        raise NotImplementedError(
            "CNN_GRU has not been implemented yet."
        )

    def forward(self, x):
        raise NotImplementedError


# ============================================================
# BiMamba
# ============================================================
class BiMamba(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
        )

        raise NotImplementedError(
            "BiMamba has not been implemented yet."
        )

    def forward(self, x):
        raise NotImplementedError


# ============================================================
# Factory
# ============================================================
def get_model(
    model_name,
    input_shape,
    num_classes,
    **kwargs,
):
    """
    Parameters
    ----------
    model_name : str
        mlp
        cnn5
        bilstm
        vit
        resnet18
        cnn_gru
        bimamba

    input_shape : tuple
        Example:
            (90,250)
            (168,250)
            (672,250)

    num_classes : int

    Returns
    -------
    nn.Module
    """

    model_name = model_name.lower()
    if model_name == "mlp":
        return MLP(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    elif model_name == "cnn5":
        return CNN5(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    elif model_name == "bilstm":
        return BiLSTM(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    elif model_name == "vit":
        return ViT(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    elif model_name == "resnet18":
        return ResNet18(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    elif model_name == "cnn_gru":
        return CNN_GRU(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    elif model_name == "bimamba":
        return BiMamba(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported model: {model_name}\n"
        f"Supported models: {SUPPORTED_MODELS}"
    )