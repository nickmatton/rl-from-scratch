# rlhf.py — Run the full RLHF pipeline: SFT → Reward Model → PPO
#
# Usage: python rlhf.py
#        python rlhf.py --skip-sft          (if you already have checkpoints/sft/)
#        python rlhf.py --skip-reward       (if you already have checkpoints/reward/)
#        python rlhf.py --ppo-only          (skip both, jump to PPO)

import argparse
import os


def run_sft():
    print("=" * 60)
    print("Stage 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)
    from algorithms.sft.sft import train
    train()
    print()


def run_reward():
    print("=" * 60)
    print("Stage 2: Reward Model Training")
    print("=" * 60)
    from models.train_reward import train_reward_model
    train_reward_model()
    print()


def run_ppo():
    print("=" * 60)
    print("Stage 3: PPO Training")
    print("=" * 60)
    from algorithms.ppo.train_ppo import main as train_ppo
    train_ppo()
    print()


def main():
    parser = argparse.ArgumentParser(description="Run the full RLHF pipeline")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT (requires checkpoints/sft/)")
    parser.add_argument("--skip-reward", action="store_true", help="Skip reward training (requires checkpoints/reward/)")
    parser.add_argument("--ppo-only", action="store_true", help="Skip SFT and reward, run PPO only")
    args = parser.parse_args()

    if args.ppo_only:
        args.skip_sft = True
        args.skip_reward = True

    # Validate that required checkpoints exist when skipping stages
    if args.skip_sft and not os.path.exists("checkpoints/sft"):
        print("Error: --skip-sft requires checkpoints/sft/ to exist. Run SFT first.")
        return
    if args.skip_reward and not os.path.exists("checkpoints/reward/reward_model.pt"):
        print("Error: --skip-reward requires checkpoints/reward/reward_model.pt to exist. Train reward model first.")
        return

    if not args.skip_sft:
        run_sft()

    if not args.skip_reward:
        run_reward()

    run_ppo()

    print("=" * 60)
    print("RLHF pipeline complete!")
    print("  SFT model:     checkpoints/sft/")
    print("  Reward model:  checkpoints/reward/")
    print("  PPO policy:    checkpoints/ppo_policy/")
    print("=" * 60)


if __name__ == "__main__":
    main()
