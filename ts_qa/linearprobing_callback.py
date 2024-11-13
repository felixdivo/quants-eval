from lightning.pytorch.callbacks import BaseFinetuning


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


class LinearProbing2Fine(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch: int = 10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`

        frozen_modules = map(
            lambda x: x[1],
            filter(lambda el: not "out_layer" in el[0], pl_module.named_modules()),
        )
        free_modules = map(
            lambda x: x[1],
            filter(lambda el: "out_layer" in el[0], pl_module.named_modules()),
        )

        self.freeze(frozen_modules)

        [m.requires_grad_() for m in free_modules]

        print(
            "Trainable parameters: {}".format(
                count_parameters(pl_module, trainable=True)
            )
        )

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        # for i, (name, param) in enumerate(pl_module.model.gpt2.named_parameters()):
        #     if 'ln' in name or 'wpe' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        if current_epoch == self._unfreeze_at_epoch:
            # unfreeze_modules = map(lambda x: x[1], filter( lambda el: "ln" in el[0] or "wpe" in el[0] or "enc_embedding" in el[0], pl_module.named_modules()))
            unfreeze_modules = map(lambda x: x[1], pl_module.named_modules())
            self.unfreeze_and_add_param_group(
                modules=unfreeze_modules,
                optimizer=optimizer,
                train_bn=True,
            )
            print(
                "Total number of parameters: {}".format(
                    count_parameters(pl_module)
                )
            )
        print(
            "Trainable parameters: {}".format(
                count_parameters(pl_module, trainable=True)
            )
        )
