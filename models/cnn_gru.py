from .base import BaseModel


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
