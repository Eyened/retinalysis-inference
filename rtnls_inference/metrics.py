# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torchmetrics import Metric


class Dice(Metric):
    def __init__(self, n_class):
        super().__init__(dist_sync_on_step=False)
        self.n_class = n_class
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros((n_class,)), dist_reduce_fx="sum")
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, preds, target, mask, loss):
        self.steps += 1
        self.dice += self.compute_stats(preds, target, mask)
        self.loss += loss

    def compute(self):
        return 100 * self.dice / self.steps, self.loss / self.steps

    def compute_stats(self, preds, target, mask):
        scores = torch.zeros(self.n_class, device=preds.device, dtype=torch.float32)
        preds = torch.argmax(preds, dim=1)

        # apply the mask to preds and target
        preds[~mask] = 0
        target[~mask] = 0

        for i in range(0, self.n_class):
            if (target != i).all():
                # no foreground class
                scores[i] += 1 if (preds != i).all() else 0
                continue
            tp, fn, fp = self.get_stats(preds, target, i)
            denom = (2 * tp + fp + fn).to(torch.float)
            class_scores = (
                (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
            )
            scores[i] += class_scores
        return scores

    @staticmethod
    def get_stats(preds, target, class_idx):
        tp = torch.logical_and(preds == class_idx, target == class_idx).sum()
        fn = torch.logical_and(preds != class_idx, target == class_idx).sum()
        fp = torch.logical_and(preds == class_idx, target != class_idx).sum()
        return tp, fn, fp
