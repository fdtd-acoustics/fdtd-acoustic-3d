from pathlib import Path
from sb3_contrib import MaskablePPO
from agent.foam_env import FoamEnv
from agent.slot_policy import SlotPolicy

env = FoamEnv(material=6, foam_size=0.25, mic_count=3, max_freq=1000)
model = MaskablePPO(SlotPolicy, env, device="cpu")

output_path = Path("../../output/agent/ppo_dummy.zip")
output_path.parent.mkdir(parents=True, exist_ok=True)
model.save(output_path)

print(f"Saved test model: {output_path}")