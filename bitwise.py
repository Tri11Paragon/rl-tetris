import network as net
from torch import nn
from config import DotDict

ACTOR_OUTPUT = 5
CRITIC_OUTPUT = 1

def network(config: DotDict, device = None):
    device = net.get_device(device)
    p = config.network.dropout
    return net.Network(
        {
            "filters": net.Lr(nn.Sequential(
                nn.Flatten(),
                nn.Linear(20, 128),
                nn.TransformerEncoderLayer(128, 8, 1024),
                nn.LayerNorm(128),
                nn.TransformerEncoderLayer(128, 8, 1024),
                nn.LayerNorm(128),
                net.Output("conv")
            ), config.network.init_lr.convLearnRate),
            "actor_head": net.Lr(nn.Sequential(
                net.Input("conv"),
                nn.Linear(128, 160),
                nn.ReLU(),
                nn.Linear(160, 40),
                nn.ReLU(),
                nn.Linear(40, ACTOR_OUTPUT),
                net.Output("action_logits")
            ), config.network.init_lr.actorLearnRate),
            "critic_head": net.Lr(nn.Sequential(
                net.Input("conv"),
                nn.Linear(128, 160),
                nn.ReLU(),
                nn.Linear(160, 40),
                nn.ReLU(),
                nn.Linear(40, CRITIC_OUTPUT),
                net.Output("state_value")
            ), config.network.init_lr.criticLearnRate)
        },
        default_lr=config.network.init_lr.convLearnRate, device=device
    )

def main():
    pass

if __name__ == "__main__":
    main()