"""MLP and factory function for action heads."""

import torch.nn as nn


class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    """Simple MLP action head for L1 regression (baseline)."""

    def __init__(self, input_dim=2048, hidden_dim=4096, action_dim=7, NUM_ACTIONS_CHUNK=8):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.model = MLPResNet(num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim)

    def predict_action(self, actions_hidden_states):
        B, T, H = actions_hidden_states.shape
        x = self.model(actions_hidden_states.reshape(B * T, H))
        return x.view(B, T, self.action_dim)

    def forward(self, actions_hidden_states):
        return self.predict_action(actions_hidden_states)


def get_action_model(config=None):
    """Build action head from config."""
    cfg = config.framework.action_model
    model_type = cfg.action_model_type
    hidden_dim = cfg.action_hidden_dim
    action_dim = cfg.action_dim
    future_win = cfg.future_action_window_size
    past_win = cfg.past_action_window_size

    if model_type == "mcf_proto":
        from mcf_proto.model.modules.action_model.MCFProto_ActionHeader import MCFProtoActionHead
        from omegaconf import ListConfig

        mcf_groups = cfg.get("mcf_groups", None)
        # backward compat: also accept tcf_groups key
        if mcf_groups is None:
            mcf_groups = cfg.get("tcf_groups", None)

        return MCFProtoActionHead(
            input_dim=hidden_dim,
            action_dim=action_dim,
            past_action_window_size=past_win,
            future_action_window_size=future_win,
            mcf_groups=mcf_groups,
            share_dictionary_across_groups=cfg.get("share_dictionary_across_groups", True),
            num_trans_prototypes=cfg.get("num_trans_prototypes", cfg.get("K", 16)),
            num_rot_prototypes=cfg.get("num_rot_prototypes", cfg.get("num_trans_prototypes", cfg.get("K", 16))),
            d=cfg.get("d", 3),
            share_R=cfg.get("share_R", True),
            share_gating=cfg.get("share_gating", True),
            share_proto_gating=cfg.get("share_proto_gating", True),
            share_z=cfg.get("share_z", None),
            hidden_dim=cfg.get("hidden_dim", None),
            entropy_mode=cfg.get("entropy_mode", "encourage_sharp"),
            rotation_output_space=cfg.get("rotation_output_space", "normalized"),
            rotation_rpy_range=(
                (float(cfg.rotation_rpy_range[0]), float(cfg.rotation_rpy_range[1]))
                if cfg.get("rotation_rpy_range", None) is not None else None
            ),
        )
    else:
        return L1RegressionActionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            action_dim=action_dim,
            NUM_ACTIONS_CHUNK=past_win + 1 + future_win,
        )
