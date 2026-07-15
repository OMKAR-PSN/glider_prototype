import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.env import GliderEnv

try:
    from stable_baselines3 import SAC
except ImportError:
    print("stable-baselines3 not installed. Cannot run SAC training.")
    exit(0)

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from collections import deque


def _curriculum_state_path(checkpoint_path: str) -> str:
    """Returns the JSON sidecar path for a given model checkpoint path."""
    base = checkpoint_path.replace(".zip", "")
    return base + "_curriculum.json"


def save_curriculum_state(checkpoint_path: str, stage: int):
    """Save curriculum stage to a small JSON file beside the checkpoint."""
    state = {"curriculum_stage": stage}
    with open(_curriculum_state_path(checkpoint_path), "w") as f:
        json.dump(state, f)


def load_curriculum_state(checkpoint_path: str) -> int:
    """Load the saved curriculum stage from a checkpoint's sidecar file.
    Returns stage 3 as a safe default if the file doesn't exist
    (so that resuming an old checkpoint never regresses to Stage 1).
    """
    sidecar = _curriculum_state_path(checkpoint_path)
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            data = json.load(f)
        stage = int(data.get("curriculum_stage", 3))
        print(f"    Loaded curriculum state: Stage {stage} (from {sidecar})")
        return stage
    print(f"    No curriculum sidecar found at {sidecar} — defaulting to Stage 3.")
    return 3


class CurriculumCallback(BaseCallback):
    """
    Curriculum Design Decision Record:
    This callback uses a rolling 50-episode success rate threshold to determine stage 
    advancement, but practically speaking, the time-ceiling failsafe (step counts) 
    is the mechanism that usually forces advancement in training. The rolling 
    threshold serves primarily as an internal diagnostic to see if the agent has 
    'solved' a stage prior to hitting the ceiling, rather than acting as the 
    sole primary trigger.
    """
    KEEP_FRACTION = 0.20  # Fraction of latest experiences to retain during flush

    def __init__(self, total_timesteps, start_stage=1, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.current_stage = start_stage
        self.success_history = deque(maxlen=50)
        
    def _on_step(self) -> bool:
        # Check for episode completions
        for done, info in zip(self.locals.get('dones', []), self.locals.get('infos', [])):
            if done:
                miss = info.get('miss_distance', float('inf'))
                is_success = (miss < 50.0)
                self.success_history.append(is_success)
                
        # Periodically report the rolling success rate so it's visible in logs
        if len(self.success_history) >= 10 and self.num_timesteps % 10000 == 0:
            rate = sum(self.success_history) / len(self.success_history)
            n = len(self.success_history)
            print(f"    [Curriculum] Step {self.num_timesteps}: trailing {n}-ep success rate = {rate*100:.1f}% "
                  f"(stage {self.current_stage}, threshold >60% over 50 eps)")
        
        force_advance = False
        new_stage = self.current_stage
        
        # Check forced ceiling
        # Stage 1 ceiling at 10% (1M steps in a 10M run) — must fire before
        # the stagnation window (6 * 500k = 3M steps) can accumulate.
        if self.current_stage == 1 and self.num_timesteps >= 0.10 * self.total_timesteps:
            force_advance = True
            new_stage = 2
            print(f"\n--- FORCED Advancement to Curriculum Stage 2 at step {self.num_timesteps} ---")
            print("    Triggered by time ceiling (10% of total budget).")
        # Stage 2 ceiling at 35% (3.5M steps in a 10M run)
        elif self.current_stage == 2 and self.num_timesteps >= 0.35 * self.total_timesteps:
            force_advance = True
            new_stage = 3
            print(f"\n--- FORCED Advancement to Curriculum Stage 3 at step {self.num_timesteps} ---")
            print("    Triggered by time ceiling (35% of total budget).")
        
        # Check rolling-average success rate
        if len(self.success_history) == 50 and not force_advance:
            success_rate = sum(self.success_history) / 50.0
            
            if success_rate > 0.60 and self.current_stage < 3:
                new_stage = self.current_stage + 1
                print(f"\n--- Advancing to Curriculum Stage {new_stage} at step {self.num_timesteps} ---")
                print(f"    Triggered by trailing 50-episode success rate: {success_rate*100:.1f}%")
                
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            
            if hasattr(self.training_env, 'env_method'):
                self.training_env.env_method('__setattr__', 'difficulty_stage', new_stage)
            else:
                self.training_env.envs[0].difficulty_stage = new_stage
            
            # Flush 80% of the replay buffer, keeping the MOST RECENT transitions.
            if hasattr(self.model, 'replay_buffer'):
                import numpy as np
                buf = self.model.replay_buffer
                old_pos = buf.pos
                old_full = buf.full
                keep_fraction = self.KEEP_FRACTION
                
                if old_full:
                    keep_count = int(buf.buffer_size * keep_fraction)
                    idx = (np.arange(old_pos - keep_count, old_pos) % buf.buffer_size)
                else:
                    keep_count = int(old_pos * keep_fraction)
                    idx = np.arange(old_pos - keep_count, old_pos)

                # Copy most-recent transitions into low indices of every buffer array
                for attr_name in ['observations', 'next_observations', 'actions',
                                  'rewards', 'dones', 'timeouts']:
                    if hasattr(buf, attr_name):
                        arr = getattr(buf, attr_name)
                        arr[:keep_count] = arr[idx]

                buf.pos = keep_count
                buf.full = False
                print(f"    Replay buffer flushed: kept {keep_count} most-recent transitions")
                
            # Clear history so we wait 50 more episodes in the new stage
            self.success_history.clear()
            
        return True

class MonteCarloEvalCallback(BaseCallback):
    def __init__(self, eval_freq=500_000, save_path="./models/checkpoints/",
                 drive_backup_path=None, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.drive_backup_path = drive_backup_path  # e.g. '/content/drive/MyDrive/glider_training/checkpoints'
        self.best_cep50 = float('inf')
        # Stagnation tracking — two-tier system:
        # Tier 1 (warn): 6 consecutive non-improving checkpoints (3M steps, 2% tolerance)
        #   → logs a warning, resets counter, keeps training.
        # Tier 2 (soft abort): 10 consecutive non-improving checkpoints (5M steps, 10% tolerance)
        #   → backstop for genuine divergence. Separate from the Stage 1 lock-in issue
        #   which is now resolved by the curriculum ceiling fix.
        self.stagnation_count = 0
        self.STAGNATION_TOLERANCE = 0.02   # 2% margin for Tier 1 warn
        self.STAGNATION_LIMIT    = 6       # Tier 1: warn after 6 consecutive non-improving
        self.DIVERGENCE_TOLERANCE = 0.10   # 10% margin for Tier 2 soft abort
        self.DIVERGENCE_LIMIT     = 10     # Tier 2: abort after 10 consecutive non-improving
        self.divergence_count = 0
        os.makedirs(save_path, exist_ok=True)
        if drive_backup_path:
            os.makedirs(drive_backup_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        import numpy as np
        # 1. NaN / Inf check on training losses every 1000 steps
        if self.n_calls % 1000 == 0:
            if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                actor_loss  = self.model.logger.name_to_value.get("train/actor_loss",  0.0)
                critic_loss = self.model.logger.name_to_value.get("train/critic_loss", 0.0)
                if np.isnan(actor_loss) or np.isnan(critic_loss) or \
                   np.isinf(actor_loss) or np.isinf(critic_loss):
                    abort_path = os.path.join(
                        self.save_path, f"sac_glider_ABORT_nan_{self.num_timesteps}_steps.zip")
                    self.model.save(abort_path)
                    print(f"\n[ABORT] NaN/Inf in loss at step {self.num_timesteps}.")
                    print(f"        Emergency checkpoint saved to {abort_path}")
                    return False

        # 2. Periodic evaluation + checkpoint
        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            model_path = os.path.join(
                self.save_path, f"sac_glider_{self.num_timesteps}_steps.zip")
            self.model.save(model_path)
            # Save curriculum stage alongside the checkpoint
            from training.train_sac import save_curriculum_state
            # Retrieve stage from the curriculum callback if available
            curriculum_stage = 3  # safe default
            for cb in self.locals.get("callback", {}).callbacks if hasattr(self.locals.get("callback", {}), "callbacks") else []:
                if hasattr(cb, "current_stage"):
                    curriculum_stage = cb.current_stage
                    break
            save_curriculum_state(model_path, curriculum_stage)
            if self.verbose > 0:
                print(f"Saved checkpoint to {model_path} (curriculum stage {curriculum_stage})")

            # Sync to Google Drive if running on Colab
            if self.drive_backup_path:
                import shutil
                drive_zip  = os.path.join(self.drive_backup_path, os.path.basename(model_path))
                drive_json = drive_zip.replace('.zip', '_curriculum.json')
                local_json = model_path.replace('.zip', '_curriculum.json')
                shutil.copy2(model_path, drive_zip)
                if os.path.exists(local_json):
                    shutil.copy2(local_json, drive_json)
                print(f"    Synced to Drive: {drive_zip}")


            print(f"\n--- Running Automatic Monte Carlo Eval at step {self.num_timesteps} ---")
            from sim.monte_carlo import run_monte_carlo
            cep50 = run_monte_carlo(num_drops=500, use_rl=True, model_path=model_path)

            # Improvement check — Tier 1 (2% tolerance)
            tier1_threshold = self.best_cep50 * (1.0 - self.STAGNATION_TOLERANCE)
            tier2_threshold = self.best_cep50 * (1.0 + self.DIVERGENCE_TOLERANCE)

            if cep50 < tier1_threshold:
                # Genuine improvement — reset both counters
                self.best_cep50 = cep50
                self.stagnation_count = 0
                self.divergence_count = 0
                print(f"    CEP50 improved to {cep50:.2f}m (best so far).")
            else:
                self.stagnation_count += 1
                print(f"    CEP50 {cep50:.2f}m did not improve by >{self.STAGNATION_TOLERANCE*100:.0f}% "
                      f"over best {self.best_cep50:.2f}m "
                      f"({self.stagnation_count}/{self.STAGNATION_LIMIT} Tier-1 consecutive).")

                # Tier 2: check for genuine divergence (10% worse than best)
                if cep50 > tier2_threshold:
                    self.divergence_count += 1
                    print(f"    [Tier-2] CEP50 {cep50:.2f}m is >{self.DIVERGENCE_TOLERANCE*100:.0f}% "
                          f"above best {self.best_cep50:.2f}m "
                          f"({self.divergence_count}/{self.DIVERGENCE_LIMIT} divergence count).")
                else:
                    self.divergence_count = 0  # variance, not divergence — reset Tier 2

                # Tier 1 action: warn and keep training
                if self.stagnation_count >= self.STAGNATION_LIMIT:
                    warn_path = os.path.join(
                        self.save_path,
                        f"sac_glider_WARN_stagnation_{self.num_timesteps}_steps.zip")
                    self.model.save(warn_path)
                    print(f"\n[WARN] Tier-1: {self.STAGNATION_LIMIT} consecutive checkpoints "
                          f"({self.STAGNATION_LIMIT * self.eval_freq / 1e6:.1f}M steps) "
                          f"without >{self.STAGNATION_TOLERANCE*100:.0f}% improvement.")
                    print(f"       Continuing training — checkpoint saved to {warn_path}")
                    self.stagnation_count = 0  # reset Tier 1, keep training

                # Tier 2 action: soft abort on genuine divergence
                if self.divergence_count >= self.DIVERGENCE_LIMIT:
                    abort_path = os.path.join(
                        self.save_path,
                        f"sac_glider_DIVERGE_{self.num_timesteps}_steps.zip")
                    self.model.save(abort_path)
                    print(f"\n[SOFT ABORT] Tier-2: {self.DIVERGENCE_LIMIT} consecutive checkpoints "
                          f">{self.DIVERGENCE_TOLERANCE*100:.0f}% above best — genuine divergence.")
                    print(f"             Final checkpoint saved to {abort_path}")
                    return False


        return True


def train_sac(total_steps=100_000, resume_path=None, drive_backup_path=None):
    import json
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    # Determine starting curriculum stage
    start_stage = 1
    if resume_path and os.path.exists(resume_path):
        start_stage = load_curriculum_state(resume_path)

    env = GliderEnv()
    env.difficulty_stage = start_stage
    print(f"Environment initialized at curriculum Stage {start_stage}.")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}")
        model = SAC.load(resume_path, env=env)
    else:
        model = SAC("MlpPolicy", env, verbose=1)
    
    # Combined eval and checkpoint callback
    eval_callback = MonteCarloEvalCallback(
        eval_freq=500_000,
        save_path="./models/checkpoints/",
        drive_backup_path=drive_backup_path,
        verbose=1,
    )
    
    # Needs the env from the model which is wrapped in a VecEnv
    curriculum_callback = CurriculumCallback(total_steps, start_stage=start_stage)
    
    callbacks = CallbackList([eval_callback, curriculum_callback])
    
    print(f"Training SAC for {total_steps} steps using Rolling-Average Curriculum + Forced Ceilings.")
    # When resuming from a checkpoint, do NOT reset the step counter —
    # otherwise learn() restarts from 0 and trains for a full extra 10M steps.
    is_resuming = resume_path is not None and os.path.exists(resume_path)
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not is_resuming,
    )
    
    model.save("models/sac_model.zip")
    print("Saved SAC model to models/sac_model.zip")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",      type=int, default=100_000, help="Total training steps")
    parser.add_argument("--resume",     type=str, default=None,    help="Path to checkpoint to resume from")
    parser.add_argument("--drive-path", type=str, default=None,    help="Google Drive checkpoint backup path (Colab only)")
    args = parser.parse_args()

    train_sac(args.steps, args.resume, args.drive_path)

