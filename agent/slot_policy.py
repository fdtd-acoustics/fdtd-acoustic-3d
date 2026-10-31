"""Policy that scores every foam slot with the same small network (Deep Sets).

A plain MlpPolicy learns "slot number 5 is good", but slot 5 is a different place in every room.
Here every slot is judged only by its features, with the same weights for all slots,
plus a summary of the whole room. Slot order and slot count do not matter.
"""
import numpy as np
import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from agent.foam_env import MAX_SLOTS, NUM_OF_FEATURES, F_EXISTS

HIDDEN = 64


class SlotNetwork(nn.Module):
    """Takes the flattened observation, returns slot scores (actor) and a room summary (critic)."""

    def __init__(self):
        super().__init__()
        self.latent_dim_pi = MAX_SLOTS  # one score per slot
        self.latent_dim_vf = HIDDEN # critic gets 64 numbers (room summary)

        """
        create one layer features -> HIDDEN, create another layer HIDDEN -> HIDDEN
        both should not negative <- ReLU
        first hidden layer creates some sort of equation with different slot features.
        Second layer is needed so we can expand even further and find correlations between multiple results of first layer
        """
        self.encoder = nn.Sequential(
            nn.Linear(NUM_OF_FEATURES, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )

        """
        we enter 2*HIDDEN numbers -> HIDDEN and then we transform it into one number - grade of a slot
        we need additional hidden layer between 2*Hidden and grade to grade slot based on the situation of a room 
        """
        self.scorer = nn.Sequential(
            nn.Linear(2 * HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1), # no ReLU because grade could say its a bad slot
        )

        """
        critic doesnt look at the slot features it looks at the room summary
        """
        self.critic = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU())

    def _encode(self, flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slots = flat_obs.view(-1, MAX_SLOTS, NUM_OF_FEATURES)           # SB3 flattens obs -> back to (batch, slots, features)
        exists = slots[..., F_EXISTS:F_EXISTS + 1]                      # (batch, slots, 1): 1 real slot, 0 padding
        slot_emb = self.encoder(slots) * exists                         # every slot through the same encoder, padding - 0
        room = slot_emb.sum(dim=1) / exists.sum(dim=1).clamp(min=1.0)   # mean over real slots only
        return slot_emb, room

    def forward_actor(self, flat_obs: torch.Tensor) -> torch.Tensor:
        slot_emb, room = self._encode(flat_obs)
        room_per_slot = room.unsqueeze(1).expand_as(slot_emb) # create a table MAX_SLOTS x room summary
        return self.scorer(torch.cat([slot_emb, room_per_slot], dim=-1)).squeeze(-1) # glues together HIDDEN Layer of encoder and room summary it returns score

    def forward_critic(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """
        calculate room summary and put it through a critic
        """
        _, room = self._encode(flat_obs)
        return self.critic(room)

    def forward(self, flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns scores and critic room summary
        """
        return self.forward_actor(flat_obs), self.forward_critic(flat_obs)


class SlotPolicy(MaskableActorCriticPolicy):
    # switch original Policy to SlotNetwork
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SlotNetwork()

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()

        self.action_net = nn.Identity() # turn off additional layer of "slot nr. 5"
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1) # create critic prediction based on the room summary

        if self.ortho_init:
            self.mlp_extractor.apply(lambda m: self.init_weights(m, gain=np.sqrt(2)))
            self.init_weights(self.mlp_extractor.scorer[-1], gain=0.01)
            self.init_weights(self.value_net, gain=1.0)

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
