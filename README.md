# Multimodal Disease-Progression Prototype 2

A small, runnable Python/PyTorch baseline aligned with PhD research on modelling chronic disease progression from multimodal health data. It is separate from the larger \`multimodal-disease-progression-demo\` repository and from PACA++.

## What it runs

- Generates a synthetic cohort with irregular visits, image-derived feature vectors, wearable summaries, and clinical/tabular variables.
- Creates a next-observed-visit stage classification task. Current visit features are used as input; the next visit's stage is only the target.
- Encodes feature groups separately and fuses their representations.
- Includes missingness flags, an auxiliary current-stage prediction head, patient-level splitting, and training-only imputation/scaling.
- Saves a model checkpoint, test metrics, and test predictions.
- Includes unit tests and a GitHub Actions test workflow.

**Important:** the initial experiment uses synthetic data only. Stage labels are artificial. It is not a clinical tool, and synthetic scores are not evidence of clinical usefulness.

## Setup

Python 3.10+ recommended.

\`\`\`bash
git clone https://github.com/sanafirdouse2323-glitch/multimodal-disease-progression-prototype-2.git
cd multimodal-disease-progression-prototype-2
python -m venv .venv
\`\`\`

Activate the environment:

**Windows PowerShell**
\`\`\`powershell
.venv\Scripts\Activate.ps1
\`\`\`

**macOS/Linux**
\`\`\`bash
source .venv/bin/activate
\`\`\`

Install dependencies and run a small smoke experiment:

\`\`\`bash
python -m pip install -r requirements.txt
python prototype.py --patients 40 --epochs 1 --batch-size 16
\`\`\`

Run a larger toy experiment:

\`\`\`bash
python prototype.py --patients 240 --epochs 12 --output-dir results
\`\`\`

Outputs:
- \`results/metrics.json\` — held-out metrics, split sizes and interpretation notes.
- \`results/test_predictions.csv\` — predicted and target stages with class probabilities.
- \`results/model.pt\` — model weights and training-only preprocessing statistics.

Run tests:

\`\`\`bash
python -m unittest discover -s tests -v
\`\`\`

## Research scope

See [docs/RESEARCH_SCOPE.md](docs/RESEARCH_SCOPE.md) for a mapping to the advertised areas and the explicit list of work not yet implemented. In this minimal version, causal treatment-effect estimation and fairness mitigation are **not implemented**; they are research directions, not claimed features.

## Scientific and data limitations

Never commit identifiable health data, scans, patient IDs or private cohort manifests. Real research requires authorised data, clinically justified targets, leakage checks, appropriate baselines, external validation and supervisor review. This model does not diagnose disease, recommend treatment or estimate an individual's real prognosis.
