"""
Custom SB3 feature extractors. Kept out of rl/train.py so rl/export_onnx.py
(and anything else that unpickles a model) can import them without pulling
in the training script and the simulator env.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TrackCNNExtractor(BaseFeaturesExtractor):
    """1D CNN over the boundary points in the obs.

    Not used by default, PPO_KWARGS is a plain MlpPolicy. To try it, put
    features_extractor_class=TrackCNNExtractor and
    features_extractor_kwargs=dict(n_ahead=...) into POLICY_KWARGS.

    Obs is outer (x, y) pairs, then inner (x, y) pairs, then speed last. Each
    wall's n_ahead+1 points are ordered closest -> furthest, so a conv with
    shared weights can pick up curvature, where an MLP just sees a flat
    unordered vector.
    """

    def __init__(self, observation_space, n_ahead: int = 20, features_dim: int = 64):
        super().__init__(observation_space, features_dim=features_dim)
        self._seq_len = n_ahead + 1

        # channels: outer_x, outer_y, inner_x, inner_y
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        cnn_out_dim = 32 * self._seq_len  # same padding, length unchanged

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 1, features_dim),  # +1 for speed
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        seq_len = self._seq_len

        boundary = obs[:, :-1]          # (batch, 4*seq_len)
        speed    = obs[:, -1:]

        # interleaved xy -> (batch, seq_len, 2) per wall
        outer = boundary[:, :2*seq_len].reshape(-1, seq_len, 2)
        inner = boundary[:, 2*seq_len:].reshape(-1, seq_len, 2)

        # Conv1d wants (batch, channels, seq_len)
        x = torch.cat([outer, inner], dim=2).permute(0, 2, 1)

        x = self.cnn(x)  # (batch, 32, seq_len)
        x = x.flatten(1)
        x = self.fc(torch.cat([x, speed], dim=1))
        return x
