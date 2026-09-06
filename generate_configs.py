"""
Configuration Generator for Federated Learning Experiments

Generates experiment grids for:
- Core performance (encryption ratio, non-IID, scalability)
- Privacy evaluation (gradient inversion, membership inference)
- Client data size sensitivity (samples_per_client)
- Ablations and generalization

Attack defaults match ProjectControl_Loop.py and can be overridden per group.
"""

import json
from pathlib import Path
from typing import Dict, Any, List


# -----------------------------
# Global toggles (safe defaults)
# -----------------------------
ENABLE_DP_CONFIGS = False      # Only enable if DP is truly implemented end-to-end
ENABLE_BYZANTINE = False       # Keep off unless implemented
OUTPUT_DIR = "./experiment_configs"


class ExperimentConfigGenerator:
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_configs: List[Dict[str, Any]] = []

    # -----------------------------
    # Helpers
    # -----------------------------
    def _add(self, cfg: Dict[str, Any]):
        # Minimal schema hygiene
        assert "group" in cfg and "exp_id" in cfg, "Each config needs group and exp_id"
        self.all_configs.append(cfg)

    def _write_group(self, group_name: str):
        group_cfgs = [c for c in self.all_configs if c["group"] == group_name]
        out_path = self.output_dir / f"{group_name}_configs.json"
        out_path.write_text(json.dumps(group_cfgs, indent=2))
        print(f"  wrote {len(group_cfgs):4d} configs -> {out_path}")

    def _write_all(self):
        out_path = self.output_dir / "all_configs.json"
        out_path.write_text(json.dumps(self.all_configs, indent=2))
        print(f"\nWrote {len(self.all_configs)} total configs -> {out_path}")

    def base(self) -> Dict[str, Any]:
        """
        Base config shared across experiments.
        Keep constant items here to avoid accidental variation.
        """
        return {
            # Core
            "num_clients": 5,
            "rounds": 2,
            "local_epochs": 5,
            "seed": 42,

            # Data/model
            "dataset": "CIFAR10",
            "model": "Lenet5",
            "nonIID": True,
            "alpha": 0.5,

            # Method knobs
            "encryption_ratio": 0.0,
            "temperature": 4.0,

            # Per-client data budget (key knob for client size experiments)
            # NOTE: if None, your DatasetLoader / splitter should default to "use all"
            "DB_samples_per_client": 2000,

            # Keep constant (do not vary)
            "local_batch_size": 128,

            # Privacy evaluation
            "attack": False,
            "attack_interval": 1,
            "run_mia": False,
            "run_dlg": False,
            "attack_seed": 2026,
            "mia_sample_size": 500,
            "mia_bootstrap_samples": 1000,
            "dlg_num_samples": 10,
            "dlg_num_restarts": 3,
            "dlg_iterations": 1000,
            "dlg_learning_rate": 0.01,
            "dlg_optimizer": "adam",
            "dlg_objective": "l2",
            "dlg_tv_weight": 0.0001,
            "dlg_early_stopping_patience": 200,
            "dlg_success_ssim": 0.5,
            "dlg_compute_lpips": False,

            # DP (optional)
            "dp_epsilon": None,
            "dp_delta": None,
        }

    # -----------------------------
    # Experiment groups
    # -----------------------------
    def generate_all_experiments(self):
        print("=" * 80)
        print("Generating Federated Learning Experiment Configurations")
        print("=" * 80)

        # Phase 1: Core Performance
        self.generate_EXP1_encryption_baselines()
        # Write outputs
        for g in [
            "EXP1"
        ]:
            self._write_group(g)
        self._write_all()

        return self.all_configs

    # EXP1
    def generate_EXP1_encryption_baselines(self):
        print("\n[EXP1] Encryption ratio sweep + baselines")

        seeds_main = [42]
        p_sweep = [0.0, 0.01, 0.03, 0.05, 0.1, 0.5, 1.0]

        # Selective encryption sweep
        for p in p_sweep:
            for seed in seeds_main:
                c = self.base()
                c.update({
                    "group": "EXP1",
                    "exp_id": f"exp1_enc_p{p}_seed{seed}",
                    "seed": seed,
                    "encryption_ratio": p,
                    "attack": True,
                    "run_mia": True,
                    "run_dlg": p < 1.0,
                })
                self._add(c)




if __name__ == "__main__":
    gen = ExperimentConfigGenerator(output_dir=OUTPUT_DIR)
    gen.generate_all_experiments()
