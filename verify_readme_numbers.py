"""Check that the headline numbers in README.md match the saved benchmark artifacts.

Run it from anywhere: python verify_readme_numbers.py
The exit code is 1 if any artifact is present but a number does not match the README.
Missing artifacts are skipped, so a fresh clone with no artifacts still passes.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(ROOT, "artifacts")

# (label, artifact file, path into the json, README value, kind)
# kind selects how the artifact value is compared to the README value:
#   pct   accuracy as a percentage to 1 decimal
#   pct2  accuracy as a percentage to 2 decimals
#   ece   expected calibration error to 3 decimals
#   size  model size in MB to 1 decimal
#   p     p-value to 2 decimals
#   sci   p-value in two-significant-figure scientific form
#   ci    a [low, high] percentage pair to 2 decimals
CHECKS = [
    ("NB before 78.0", "evaluation_results_full25k_pre-step8.json",
     ["results", "naive_bayes_raw", "accuracy"], 78.0, "pct"),
    ("LR before 80.2", "evaluation_results_full25k_pre-step8.json",
     ["results", "logistic_regression_raw", "accuracy"], 80.2, "pct"),

    ("NB after 85.5", "benchmark_multi.json",
     ["imdb", "components", "naive_bayes", "accuracy"], 85.5, "pct"),
    ("LR after 90.1", "benchmark_multi.json",
     ["imdb", "components", "logistic_regression", "accuracy"], 90.1, "pct"),
    ("LinearSVC 90.1", "benchmark_multi.json",
     ["imdb", "components", "linear_svc", "accuracy"], 90.1, "pct"),
    ("NBSVM IMDB 90.8", "benchmark_multi.json",
     ["imdb", "components", "nbsvm", "accuracy"], 90.8, "pct"),

    ("off-the-shelf DistilBERT IMDB 89.1", "benchmark_multi.json",
     ["imdb", "components", "distilbert", "accuracy"], 89.1, "pct"),
    ("off-the-shelf DistilBERT SST-2 91.1", "benchmark_multi.json",
     ["sst2", "components", "distilbert", "accuracy"], 91.1, "pct"),
    ("off-the-shelf DistilBERT Yelp 91.5", "benchmark_multi.json",
     ["yelp", "components", "distilbert", "accuracy"], 91.5, "pct"),
    ("VADER IMDB 70.0", "benchmark_multi.json",
     ["imdb", "components", "vader", "accuracy"], 70.0, "pct"),
    ("VADER SST-2 66.2", "benchmark_multi.json",
     ["sst2", "components", "vader", "accuracy"], 66.2, "pct"),
    ("VADER Yelp 72.0", "benchmark_multi.json",
     ["yelp", "components", "vader", "accuracy"], 72.0, "pct"),
    ("NBSVM SST-2 79.1", "benchmark_multi.json",
     ["sst2", "components", "nbsvm", "accuracy"], 79.1, "pct"),
    ("NBSVM Yelp 86.0", "benchmark_multi.json",
     ["yelp", "components", "nbsvm", "accuracy"], 86.0, "pct"),

    ("fine-tuned DistilBERT IMDB 91.2", "stack_results.json",
     ["imdb", "components", "distilbert", "accuracy"], 91.2, "pct"),
    ("fine-tuned DistilBERT SST-2 84.4", "stack_results.json",
     ["sst2", "components", "distilbert", "accuracy"], 84.4, "pct"),
    ("fine-tuned DistilBERT Yelp 91.3", "stack_results.json",
     ["yelp", "components", "distilbert", "accuracy"], 91.3, "pct"),
    ("stack IMDB 93.0", "stack_results.json",
     ["imdb", "components", "stack", "accuracy"], 93.0, "pct"),
    ("stack IMDB 92.98", "stack_results.json",
     ["imdb", "components", "stack", "accuracy"], 92.98, "pct2"),
    ("stack IMDB CI 92.66 to 93.29", "stack_results.json",
     ["imdb", "components", "stack", "acc_ci95"], [92.66, 93.29], "ci"),
    ("stack vs best McNemar p 2.71e-33", "stack_results.json",
     ["imdb", "mcnemar_stack_vs_best", "p_value"], 2.71e-33, "sci"),
    ("stack SST-2 85.7", "stack_results.json",
     ["sst2", "components", "stack", "accuracy"], 85.7, "pct"),
    ("stack SST-2 p 0.21", "stack_results.json",
     ["sst2", "mcnemar_stack_vs_best", "p_value"], 0.21, "p"),
    ("stack Yelp 92.0", "stack_results.json",
     ["yelp", "components", "stack", "accuracy"], 92.0, "pct"),
    ("stack Yelp p 0.26", "stack_results.json",
     ["yelp", "mcnemar_stack_vs_best", "p_value"], 0.26, "p"),

    ("DistilBERT long-review 87.5", "error_analysis.json",
     ["models", "distilbert", "length", "long(>150w)", "acc"], 87.5, "pct"),
    ("NBSVM long-review 88.6", "error_analysis.json",
     ["models", "nbsvm", "length", "long(>150w)", "acc"], 88.6, "pct"),
    ("stack long-review 89.8", "error_analysis.json",
     ["models", "stack", "length", "long(>150w)", "acc"], 89.8, "pct"),
    ("fine-tuned DistilBERT 2k overall 89.6", "error_analysis.json",
     ["models", "distilbert", "overall", "acc"], 89.6, "pct"),

    ("zero-shot distilbert-mnli 71.8", "zero_shot_baseline.json",
     ["models", "typeform/distilbert-base-uncased-mnli", "accuracy"], 71.8, "pct"),
    ("zero-shot bart-large-mnli 87.7", "zero_shot_baseline.json",
     ["models", "facebook/bart-large-mnli", "accuracy"], 87.7, "pct"),

    ("ECE LR 0.020", "calibration.json", ["ece", "logistic_regression"], 0.020, "ece"),
    ("ECE LinearSVC 0.021", "calibration.json", ["ece", "linear_svc"], 0.021, "ece"),
    ("ECE NBSVM 0.021", "calibration.json", ["ece", "nbsvm"], 0.021, "ece"),
    ("ECE DistilBERT 0.032", "calibration.json", ["ece", "distilbert"], 0.032, "ece"),
    ("ECE stack 0.031", "calibration.json", ["ece", "stack"], 0.031, "ece"),

    ("size NBSVM 12.2 MB", "cost_table.json", ["options", "nbsvm", "size_mb"], 12.2, "size"),
    ("size LinearSVC 2.4 MB", "cost_table.json", ["options", "linear_svc", "size_mb"], 2.4, "size"),
    ("size DistilBERT 268.5 MB", "cost_table.json", ["options", "distilbert", "size_mb"], 268.5, "size"),
    ("size rule-based 22.6 MB", "cost_table.json", ["options", "rule_based", "size_mb"], 22.6, "size"),
]


def dig(data, path):
    for key in path:
        data = data[key]
    return data


def matches(value, expected, kind):
    if kind == "pct":
        return abs(value * 100 - expected) <= 0.051
    if kind == "pct2":
        return abs(value * 100 - expected) <= 0.0051
    if kind == "ece":
        return abs(value - expected) <= 0.00051
    if kind == "size":
        return abs(value - expected) <= 0.051
    if kind == "p":
        return abs(value - expected) <= 0.0051
    if kind == "sci":
        return "%.2e" % value == "%.2e" % expected
    if kind == "ci":
        return (abs(value[0] * 100 - expected[0]) <= 0.0051 and
                abs(value[1] * 100 - expected[1]) <= 0.0051)
    raise ValueError("unknown kind: " + kind)


def shown(value, kind):
    if kind == "pct":
        return "%.1f" % (value * 100)
    if kind == "pct2":
        return "%.2f" % (value * 100)
    if kind == "ece":
        return "%.3f" % value
    if kind == "size":
        return "%.1f" % value
    if kind == "p":
        return "%.2f" % value
    if kind == "sci":
        return "%.2e" % value
    if kind == "ci":
        return "[%.2f, %.2f]" % (value[0] * 100, value[1] * 100)
    return str(value)


def main():
    cache = {}
    passed = 0
    failed = 0
    skipped = 0
    bad = []
    for label, fname, path, expected, kind in CHECKS:
        if fname not in cache:
            full = os.path.join(ARTIFACTS, fname)
            cache[fname] = json.load(open(full)) if os.path.exists(full) else None
        data = cache[fname]
        if data is None:
            print("SKIP  %s (artifact %s not found)" % (label, fname))
            skipped += 1
            continue
        try:
            value = dig(data, path)
            ok = matches(value, expected, kind)
        except (KeyError, TypeError, IndexError):
            print("FAIL  %s (could not read the value in %s)" % (label, fname))
            failed += 1
            bad.append(label)
            continue
        if ok:
            print("PASS  %s (readme %s, artifact %s)" % (label, expected, shown(value, kind)))
            passed += 1
        else:
            print("FAIL  %s (readme %s, artifact %s)" % (label, expected, shown(value, kind)))
            failed += 1
            bad.append(label)

    print()
    print("%d passed, %d failed, %d skipped" % (passed, failed, skipped))
    if failed:
        print("mismatches: " + ", ".join(bad))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
