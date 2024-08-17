from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import fullname


class SigmoidLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss_fct: nn.Module = nn.BCELoss(),
        cos_score_transformation: nn.Module = nn.Sigmoid(),
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor) -> torch.Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        dot_product = torch.sum(embeddings[0] * embeddings[1], dim=1)
        output = self.cos_score_transformation(dot_product)
        return self.loss_fct(output, labels.float().view(-1))

    def get_config_dict(self) -> dict[str, Any]:
        return {"loss_fct": fullname(self.loss_fct)}
