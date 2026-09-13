import json

try:
    with open('crossval_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading JSON: {e}")
    exit(1)

summary = data.get("summary", {})

# Prepare the new section text
new_section = f"""## 4. M5: 5-Fold Cross-Validation with Early Stopping

5-fold stratified cross-validation with early stopping (patience=10) validates that model performance is not an artifact of a single train/test split.

**Complete 5-Fold Aggregated Results**:

| Metric | Mean ± Std |
|---|---:|
| AUC | {summary.get('auc', {}).get('mean', 0):.4f} ± {summary.get('auc', {}).get('std', 0):.4f} |
| Accuracy | {summary.get('accuracy', {}).get('mean', 0):.4f} ± {summary.get('accuracy', {}).get('std', 0):.4f} |
| Precision | {summary.get('precision', {}).get('mean', 0):.4f} ± {summary.get('precision', {}).get('std', 0):.4f} |
| Recall | {summary.get('recall', {}).get('mean', 0):.4f} ± {summary.get('recall', {}).get('std', 0):.4f} |
| F1 | {summary.get('f1', {}).get('mean', 0):.4f} ± {summary.get('f1', {}).get('std', 0):.4f} |

> Results saved to `crossval_results.json` and `training_curves.png`.

### Training Configuration
- **Max epochs**: 20 (with early stopping, patience=10)
- **Validation split**: 20% of training data per fold
- **Test split**: Fixed 20% holdout per fold
- **Stratification**: Preserves class ratio across all splits
"""

try:
    with open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()

    start_str = "## 4. M5: 5-Fold Cross-Validation with Early Stopping"
    end_str = "## 5. M6: Class Weighting Ablation"
    
    start_idx = readme.find(start_str)
    end_idx = readme.find(end_str)
    
    if start_idx != -1 and end_idx != -1:
        new_readme = readme[:start_idx] + new_section + "\n" + readme[end_idx:]
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(new_readme)
        print("README.md updated successfully with Cross-Validation results!")
    else:
        print("Could not find the target section in README.md")
except Exception as e:
    print(f"Error updating README.md: {e}")
