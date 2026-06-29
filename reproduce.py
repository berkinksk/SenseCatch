"""Regenerate every benchmark artifact the README cites, in dependency order.

This is the single entry point for reproducing the results. It runs the existing
scripts one after another with the current Python interpreter and streams their
output live. The full run is long: training the classical models takes about 40
minutes on a cold cache, and fine-tuning the transformer is slow without a GPU.

Usage:
    python reproduce.py              run the whole pipeline
    python reproduce.py --dry-run    print the stages and exact commands, run nothing
    python reproduce.py --skip-train skip the three training stages and only
                                     regenerate the evaluation, analysis and
                                     verification artifacts (the models must already exist)
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# (label, command arguments relative to the repo root, is_training_stage)
STAGES = [
    ("data split", ["src/training/data_split.py"], False),
    ("train the 4 classical models", ["src/training/retrain.py", "--train-deployed"], True),
    ("fine-tune the transformer", ["src/training/finetune_distilbert.py"], True),
    ("build the stacked ensemble", ["src/training/stack_ensemble.py"], True),
    ("multi-dataset benchmark", ["benchmark.py"], False),
    ("rule diagnosis", ["src/training/diagnose_rules.py"], False),
    ("calibration (ECE)", ["src/training/calibration.py"], False),
    ("error analysis", ["src/training/error_analysis.py"], False),
    ("cost table", ["src/training/cost_table.py"], False),
    ("zero-shot baselines", ["src/training/zero_shot_baseline.py"], False),
    ("verify the numbers", ["verify_readme_numbers.py"], False),
]


def build_command(args):
    """Turn a stage's argument list into a full command using this interpreter."""
    script = os.path.join(ROOT, args[0])
    return [sys.executable, script] + args[1:]


def main():
    parser = argparse.ArgumentParser(description="Regenerate the benchmark artifacts in order.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the stages and commands without running anything")
    parser.add_argument("--skip-train", action="store_true",
                        help="skip the training stages and only run eval, analysis and verify")
    opts = parser.parse_args()

    stages = [s for s in STAGES if not (opts.skip_train and s[2])]

    if opts.dry_run:
        print("Planned stages (%d):" % len(stages))
        for i, (label, args, _) in enumerate(stages, 1):
            print("  %2d. %-28s %s" % (i, label, " ".join(build_command(args))))
        return

    total = len(stages)
    for i, (label, args, _) in enumerate(stages, 1):
        command = build_command(args)
        print("\n=== stage %d/%d: %s ===" % (i, total, label), flush=True)
        print("    " + " ".join(command), flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            print("\nStage %d (%s) failed with exit code %d." % (i, label, result.returncode))
            raise SystemExit(result.returncode)

    print("\nAll %d stages finished." % total)


if __name__ == "__main__":
    main()
