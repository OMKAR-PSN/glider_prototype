"""
Sanity Run: 100k steps with the eval_freq patched to 50k.
This is the cleanest way to inject the param without modifying train_sac.py.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from training.env import GliderEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from training.train_sac import CurriculumCallback, MonteCarloEvalCallback

TOTAL_STEPS = 100_000

env = GliderEnv()
env.difficulty_stage = 1
print(f"Environment initialized at curriculum Stage 1.")

model = SAC("MlpPolicy", env, verbose=1)

eval_callback = MonteCarloEvalCallback(
    eval_freq=50_000,         # Fire at 50k and 100k — guarantees one real eval
    save_path="./models/sanity_checkpoints/",
    drive_backup_path=None,
    verbose=1,
)

curriculum_callback = CurriculumCallback(TOTAL_STEPS, start_stage=1)
callbacks = CallbackList([eval_callback, curriculum_callback])

print(f"Training SAC for {TOTAL_STEPS} steps (SANITY RUN — 20Hz fix verification).")
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=callbacks,
    progress_bar=True,
    reset_num_timesteps=True,
)
model.save("models/sanity_checkpoints/sac_sanity_final.zip")
print("Sanity run complete. Final model saved.")
