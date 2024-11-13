from typing import TypedDict, cast
import torch
from torchmetrics import Accuracy, MetricCollection

from .conv_positional_encoding import ConvPositionalEncoding

from .grad_multiply import GradMultiply
from .classification_system import ClassificationModelSystem
from lightning import LightningModule
import torch.nn as nn
from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    PatchTSMixerForTimeSeriesClassification,
    PatchTSMixerModel,
    AutoModel,
    AutoTokenizer,
    PatchTSMixerForPretraining,
)
from transformers.models.patchtsmixer.modeling_patchtsmixer import (
    PatchTSMixerModelOutput,
    PatchTSMixerForPredictionOutput,
    PatchTSMixerForPreTrainingOutput,
)
from einops import rearrange, repeat


Item = tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]

class ItemDict(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    answer: torch.LongTensor
    trajectory: torch.FloatTensor


class MMTransformerPretrain(LightningModule):
    def __init__(
        self,
        seq_len: int,
        patch_len: int = 4,
        patch_stride: int = 4,
        feat_dim: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.config = PatchTSMixerConfig(
            context_length=seq_len,
            num_input_channels=feat_dim,
            patch_length=patch_len,
            patch_stride=patch_stride,
            return_loss=True,
            mode="mix_channel",
            d_model=768,
            use_positional_encoding=True,
            positional_encoding_type="sincos",
        )

        self.time_extractor = PatchTSMixerForPretraining(self.config)
        self.lr = lr

    def training_step(self, batch: Item, batch_idx):
        _, _, inputs, labels = batch
        inputs = inputs.transpose(1, 2)  # batch x seq x feat
        outputs: PatchTSMixerForPreTrainingOutput = self.time_extractor(
            inputs, inputs.clone()
        )
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Item, batch_idx):
        _, _, inputs, labels = batch
        inputs = inputs.transpose(1, 2)
        outputs: PatchTSMixerForPreTrainingOutput = self.time_extractor(
            inputs, inputs.clone()
        )
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.lr)


class MMTransformer(MMTransformerPretrain):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        time_grad_mul: float = 1.0,
        feat_dim: int = 1,
        tok_len: int = 30,
        patch_len: int = 4,
        patch_stride: int = 4,
        lr: float = 1e-4,
    ):
        super().__init__(
            seq_len=seq_len,
            feat_dim=feat_dim,
            lr=lr,
            patch_len=patch_len,
            patch_stride=patch_stride,
        )

        self.time_extractor = PatchTSMixerModel(self.config)
        self.num_patches = self.time_extractor.patching.num_patches
        self.time_dim = self.num_patches * feat_dim  # 888
        self.feat_dim = feat_dim
        self.tok_len = tok_len

        self.feat_layer_norm = nn.LayerNorm(feat_dim)
        self.text_layer_norm = nn.LayerNorm(768)  # bert embedding_dim
        self.dropout_features = nn.Dropout(
            0.0
        )  # dropout to apply to the features (after feat extr)
        self.time_grad_mul = time_grad_mul
        self.lr = lr
        # self.time_pos_embedding = self.time_extractor.get_position_embeddings()

        # if from pretrained
        bert_embeddings = AutoModel.from_pretrained("bert-base-uncased").embeddings
        self.language_embedding = bert_embeddings.word_embeddings
        self.position_embedding = bert_embeddings.position_embeddings
        self.token_type_embedding = bert_embeddings.token_type_embeddings

        # self.time_token_type_embedding = nn.Embedding(
        #         feat_dim, 768
        #     )

        # else https://github.dev/Jwoo5/fairseq-signals/blob/master/fairseq_signals/models/ecg_language_transformer.py#L13

        # conv_pos = 128  # number of filters for convolutional positional embeddings
        # conv_pos_groups = 16  # number of groups for convolutional positional embeddings
        # # self.conv_pos = ConvPositionalEncoding(768, conv_pos, conv_pos_groups)  # todo

        # # self.post_extract_proj = nn.Linear(160, 768)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=2, # 12
            dim_feedforward=32,#3072,
            activation="gelu",
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Linear(768 * (feat_dim + 1), num_classes)

        self.crit = nn.CrossEntropyLoss()

        metrics = MetricCollection(
            [Accuracy(task="multiclass", num_classes=num_classes)]
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def process_text_features(self, text_ids):
        text_features = self.language_embedding(text_ids)
        text_features = self.text_layer_norm(text_features)
        text_features = self.dropout_features(text_features)

        indices = torch.arange(text_features.size(1), device=text_features.device)
        repeated_indices = repeat(indices, "t -> b t", b=text_features.size(0))
        text_features_pos = self.position_embedding(repeated_indices)

        text_features = text_features + text_features_pos
        text_features_type_embedding = self.token_type_embedding(
            text_features.new_zeros(text_features.shape[:-1], dtype=torch.int32)
        )
        text_features = text_features + text_features_type_embedding

        return text_features

    def process_time_features(self, x):
        x = x
        time_features_output = self.time_extractor(x)
        time_features = time_features_output.last_hidden_state

        if self.time_grad_mul != 1.0:
            time_features = GradMultiply.apply(time_features, self.time_grad_mul)

        time_features = self.feat_layer_norm(time_features.transpose(1, 3))
        time_features = self.dropout_features(time_features).transpose(1, 3)

        time_token_type_embedding = self.token_type_embedding(
            time_features.new_full(time_features.shape[:-1], 1, dtype=torch.int32)
        )
        time_features = time_features + time_token_type_embedding

        return rearrange(time_features, "b var n p -> b (var n) p")

    # def forward(self, x, text_ids, text_mask):
    #     # todo add padding mask impl

    #     x = x.transpose(1, 2)
    #     # batch x seq x var
    #     time_features_output: PatchTSMixerModelOutput = self.time_extractor(x)
    #     time_features = time_features_output.last_hidden_state
    #     # b x var x num_patches x patch_dim

    #     # todo check if we need a projection
    #     # time_features = rearrange(time_features, "b var pnum pwidth -> b var (pnum pwidth)")
    #     # time_features = self.post_extract_proj(time_features)
    #     # time_features = rearrange(time_features, "b var t -> b t var")
    #     # should be (B, T, C) if not transpose

    #     if self.time_grad_mul != 1.0:
    #         time_features = cast(
    #             torch.Tensor, GradMultiply.apply(time_features, self.time_grad_mul)
    #         )

    #     # feature normalization (Channel)
    #     time_features: torch.Tensor = self.feat_layer_norm(
    #         time_features.transpose(1, 3)
    #     )
    #     time_features: torch.Tensor = self.dropout_features(time_features)
    #     time_features = time_features.transpose(1, 3)

    #     b, variates, num_patches, patch_dim = time_features.shape
    #     #  # Apply positional encoding to time patches
    #     # time_indices = torch.arange(num_patches, device=time_features.device)
    #     # time_repeated_indices = repeat(time_indices, 'n -> (b var) n', b=b, var=variates)
    #     # time_pos_embedding = self.time_pos_embedding(time_repeated_indices)
    #     # time_pos_embedding = rearrange(time_pos_embedding, '(b var) n p -> b var n p', b=b, var=variates)
    #     # time_features = time_features + time_pos_embedding

    #     # # Add token type embedding to time patches
    #     time_token_type_embedding = self.token_type_embedding(
    #         time_features.new_full(time_features.shape[:-1], 1, dtype=torch.int32)
    #     )
    #     time_features = time_features + time_token_type_embedding

    #     # # Add token type embedding to time feature patches, treating each variate as a separate segment
    #     # time_token_type_embeddings = []
    #     # for c in range(variates):
    #     #     token_type_embedding = self.time_token_type_embedding(
    #     #         time_features.new_full((b, num_patches), c, dtype=torch.int32)  # c + 2 to distinguish from other modalities
    #     #     )
    #     #     time_token_type_embeddings.append(token_type_embedding.unsqueeze(1))
    #     # time_token_type_embedding = torch.cat(time_token_type_embeddings, dim=1)
    #     # time_features = time_features + time_token_type_embedding

    #     # time_features = time_features + time_token_type_embedding

    #     # Combine all features (flatten the time feature patches)
    #     time_features = rearrange(time_features, "b var n p -> b (var n) p")

    #     # time_features_pos: torch.Tensor = self.conv_pos(
    #     #     time_features, channel_first=False
    #     # )
    #     # time_features = time_features + time_features_pos

    #     # time_features_type_embedding = self.token_type_embedding(
    #     #     time_features.new_zeros(time_features.shape[:-1], dtype=torch.int32)
    #     # ) # embed time_features, but we skip the last dim
    #     # # Creates a token type embedding for the time features.
    #     # # The type embedding needs to be applied to each timestep in the sequence, not each feature.
    #     # # Thus, the tensor of zeros with shape ( 𝐵 , 𝑇)
    #     # # (omitting the feature dimension) is passed to the embedding layer.

    #     # time_features = time_features + time_features_type_embedding

    #     # text_features
    #     text_features = self.language_embedding(text_ids)
    #     text_features = self.text_layer_norm(text_features)
    #     text_features = self.dropout_features(text_features)

    #     # Create the range of indices for positional embedding
    #     indices = torch.arange(text_features.size(1), device=text_features.device)

    #     # Repeat the indices for each item in the batch
    #     repeated_indices = repeat(indices, "t -> b t", b=text_features.size(0))

    #     # Apply positional embedding
    #     text_features_pos = self.position_embedding(repeated_indices)

    #     text_features: torch.Tensor = text_features + text_features_pos
    #     text_features_type_embedding = self.token_type_embedding(
    #         text_features.new_zeros(text_features.shape[:-1], dtype=torch.int32)
    #     )
    #     text_features = text_features + text_features_type_embedding

    #     # concat features
    #     features = torch.cat([text_features, time_features], dim=1)
    #     # token batch x 40 x 768
    #     # batch X 480 x 768

    #     encoding = self.encoder(features)

    #     text_features = encoding[:, : text_features.size(1)]
    #     text_features = torch.div(
    #         text_features.sum(dim=1), (text_features != 0).sum(dim=1)
    #     )

    #     time_features = encoding[:, -time_features.size(1) :]
    #     # Normalize time features while keeping individual variates
    #     # non_zero_counts_time = (time_features != 0).sum(dim=0, keepdim=True)  # Count non-zeros for each variate across the batch
    #     # time_features = torch.div(time_features, non_zero_counts_time.where(non_zero_counts_time != 0, torch.tensor(1.0)))
    #     time_features = rearrange(
    #         time_features, "b (var n) p -> b var n p", var=variates
    #     )
    #     time_features = torch.div(
    #         time_features.sum(dim=2), (time_features != 0).sum(dim=2)
    #     )
    #     time_features = torch.zeros_like(time_features)
    #     features = torch.cat([text_features.unsqueeze(1), time_features], dim=1)
    #     # flatten
    #     features = features.view(features.size(0), -1)

    #     return self.classifier(features)
    def forward(self, x, text_ids, text_mask):
        time_features = self.process_time_features(x)
        text_features = self.process_text_features(text_ids)

        # Combine features
        features = torch.cat([text_features, time_features], dim=1)
        encoding = self.encoder(features)

        text_features = encoding[:, : text_features.size(1)]
        text_features = torch.div(
            text_features.sum(dim=1), (text_features != 0).sum(dim=1)
        )

        time_features = encoding[:, -time_features.size(1) :]
        time_features = rearrange(
            time_features, "b (var n) p -> b var n p", var=self.feat_dim
        )
        time_features = torch.div(
            time_features.sum(dim=2), (time_features != 0).sum(dim=2)
        )
        features = torch.cat([text_features.unsqueeze(1), time_features], dim=1)
        features = features.view(features.size(0), -1)

        return self.classifier(features)

    def on_train_epoch_end(self):
        # log epoch metric
        self.log_dict(self.train_metrics, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics, prog_bar=True)

    def training_step(self, batch: ItemDict, batch_idx):
        text_ids, text_mask, x, y = batch["input_ids"], batch["attention_mask"], batch["trajectory"], batch["answer"]
        outputs = self(x, text_ids, text_mask)
        loss = self.crit(outputs, y)
        self.log("train/loss", loss, prog_bar=True)

        # loss = self.criterion(outputs, y)
        self.train_metrics(outputs, y)
        # return loss
        return loss

    def validation_step(self, batch: ItemDict, batch_idx):
        text_ids, text_mask, x, y = batch["input_ids"], batch["attention_mask"], batch["trajectory"], batch["answer"]
        outputs = self(x, text_ids, text_mask)

        loss = self.crit(outputs, y)
        self.log("val/loss", loss, prog_bar=True)
        self.val_metrics(outputs, y)
        return loss

    def configure_optimizers(self):
        # print("Total number of parameters: {}".format(count_parameters(self.model)))
        # print(
        #     "Trainable parameters: {}".format(
        #         count_parameters(self.model, trainable=True)
        #     )
        # )
        # optimizer = torch.optim.RAdam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        optimizer = torch.optim.RAdam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        return optimizer

    def freeze(self):
        for param in self.time_extractor.parameters():
            param.requires_grad = False
        for param in self.language_embedding.parameters():
            param.requires_grad = False
        for param in self.position_embedding.parameters():
            param.requires_grad = False
        for param in self.token_type_embedding.parameters():
            param.requires_grad = False

    def load_pretrained(self, path: str):
        state_dict = torch.load(path)["state_dict"]
        time_extractor_state_dict = {
            k.replace("time_extractor.model.", ""): v
            for k, v in state_dict.items()
            if "time_extractor" in k
        }
        # filter head key from state_dict
        time_extractor_state_dict = {
            k: v for k, v in time_extractor_state_dict.items() if "head" not in k
        }
        missing_info = self.time_extractor.load_state_dict(
            time_extractor_state_dict, strict=True
        )
        for k in missing_info.missing_keys:
            print(f"Missing key: {k}")


class TextOnlyTransformer(MMTransformer):
    def forward(self, x, text_ids, text_mask):
        text_features = self.process_text_features(text_ids)
        features = torch.cat(
            [
                text_features,
                torch.zeros(
                    (text_features.shape[0], self.time_dim, text_features.shape[-1]),
                    device=text_features.device,
                ),
            ],
            dim=1,
        )
        encoding = self.encoder(features)

        text_features = encoding[:, : text_features.size(1)]
        text_features = torch.div(
            text_features.sum(dim=1), (text_features != 0).sum(dim=1)
        )

        time_features = encoding[:, self.tok_len :]
        time_features = rearrange(
            time_features, "b (var n) p -> b var n p", var=self.feat_dim
        )

        time_features = torch.div(
            time_features.sum(dim=2), (time_features != 0).sum(dim=2)
        )

        features = torch.cat([text_features.unsqueeze(1), time_features], dim=1)
        features = features.view(features.size(0), -1)

        return self.classifier(features)


class TimeOnlyTransformer(MMTransformer):
    def forward(self, x, text_ids=None, text_mask=None):
        time_features = self.process_time_features(x)

        features = torch.cat(
            [
                torch.zeros(
                    (time_features.shape[0], self.tok_len, time_features.shape[-1]),
                    device=time_features.device,
                ),
                time_features,
            ],
            dim=1,
        )
        encoding = self.encoder(features)

        text_features = encoding[:, : self.tok_len]
        text_features = torch.div(
            text_features.sum(dim=1), (text_features != 0).sum(dim=1)
        )

        time_features = encoding[:, -time_features.size(1) :]
        time_features = rearrange(
            time_features, "b (var n) p -> b var n p", var=self.feat_dim
        )
        time_features = torch.div(
            time_features.sum(dim=2), (time_features != 0).sum(dim=2)
        )
        features = torch.cat([text_features.unsqueeze(1), time_features], dim=1)
        features = features.view(features.size(0), -1)

        return self.classifier(features)
