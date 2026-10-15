# Copyright (c) 2026 PaddlePaddle Authors and Turkish SynthDoG contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keep PPLCNetV4's inference width during long-line recognition training.

The pinned upstream training forward pools every width to 40 time steps.
Use its existing inference pooling for training too. All learnable layers,
state-dict keys, BatchNorm training behavior and inference behavior stay intact.
Derived from PaddleOCR ppocr/modeling/backbones/rec_lcnetv4.py (Apache-2.0).
"""


def install_width_preserving_pool(config):
    if not config.get("TurkishRecognitionPooling", {}).get("enabled"):
        raise ValueError("Long-line v6 training requires width-preserving pooling")
    from ppocr.modeling.backbones.rec_lcnetv4 import PPLCNetV4
    from paddle.nn import functional as F

    if getattr(PPLCNetV4.forward, "turkish_width_preserving", False):
        return
    original = PPLCNetV4.forward

    def forward(self, x):
        if self.det or not self.training:
            return original(self, x)
        for name in ("conv1", "blocks2", "blocks3", "blocks4", "blocks5", "blocks6"):
            x = getattr(self, name)(x)
        if x.shape[2] < 3:
            raise ValueError("Use the validated 48-pixel recognition height")
        return F.avg_pool2d(x, [3, 2])

    forward.turkish_width_preserving = True
    PPLCNetV4.forward = forward
