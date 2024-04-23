import torch
from .gpt4ts import GPT4ts
from .classification_system import ClassificationModelSystem
import torch.nn as nn


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


class OFA(ClassificationModelSystem):
    def __init__(
        self,
        max_token_length: int,
        num_classes: int,
        max_seq_len: int,
        patch_size: int,
        stride: int,
        dropout: float,
        d_model: int = 768,
        feat_dim: int = 1,
        
    ):
        super().__init__(
            num_classes=num_classes,
        )

        self.model = GPT4ts(
            max_token_length,
            max_seq_len, patch_size, stride, dropout, num_classes, d_model, feat_dim
        )
        # self.backbone = Backbone(input_size=1)
        # self.classifier = nn.LazyLinear(num_classes)

        # for name, param in self.model.named_parameters():
        #     if name.startswith("output_layer"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        print("Model:\n{}".format(self.model))
        print("Total number of parameters: {}".format(count_parameters(self.model)))
        print(
            "Trainable parameters: {}".format(
                count_parameters(self.model, trainable=True)
            )
        )

    def forward(self, x, text):
        # batch x length x feats
        out = self.model(x.transpose(1, 2), text)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.0005)
        return optimizer
