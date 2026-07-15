import torch
import numpy as np
try:
    from stable_baselines3 import SAC
except ImportError:
    print("stable-baselines3 not installed")
    exit(0)

def export_onnx(model_path="models/sac_model.zip", out_path="models/sac_policy.onnx"):
    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print(f"SAC model not found at {model_path}")
        return

    # Set to eval mode before export to avoid training-mode artifacts
    model.policy.set_training_mode(False)

    class OnnxablePolicy(torch.nn.Module):
        def __init__(self, actor):
            super(OnnxablePolicy, self).__init__()
            self.actor = actor

        def forward(self, observation):
            features = self.actor.features_extractor(observation)
            latent_pi = self.actor.latent_pi(features)
            mean_actions = self.actor.mu(latent_pi)
            # SAC deterministic action is just tanh of the mean
            return torch.tanh(mean_actions)

    onnxable_model = OnnxablePolicy(model.policy.actor)
    onnxable_model.eval()  # Ensure inference mode before export

    dummy_input = torch.randn(1, model.observation_space.shape[0])  # Match env obs dim
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        out_path,
        opset_version=18,
        input_names=["observation"],
        output_names=["action"]
    )
    print(f"Exported SAC policy to ONNX format at {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export SAC model to ONNX")
    parser.add_argument("--model", type=str, default="models/sac_model.zip", help="Path to input .zip model")
    parser.add_argument("--out", type=str, default="models/sac_policy.onnx", help="Path to output .onnx file")
    args = parser.parse_args()
    
    export_onnx(args.model, args.out)
