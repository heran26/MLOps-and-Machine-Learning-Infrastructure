# ML CI/CD Pipeline (Train -> Evaluate -> Auto-Deploy)

An automated pipeline that:
1. Retrains a model (scheduled, on-demand, or on code push)
2. Evaluates it against a fixed held-out test set
3. Promotes + deploys it **only if accuracy improves** over the current production model

## Important: where things actually run

- **Training in the automated pipeline runs inside the GitHub Actions runner** -- a cloud
  machine GitHub provides for free (2 CPU, 7GB RAM), not your PC. This is what solves your
  "my machine crashes" problem for the *automated* retraining loop.
- **Google Colab** is for the parts GitHub Actions can't do well: interactive experimentation,
  bigger models, or free GPU access. Colab's free tier has no reliable way to be triggered
  headlessly on a schedule, so it isn't part of the automated loop itself -- you use it to
  develop/tune `src/train.py`, then let the committed script run automatically in CI.
  If you later need GPU training *inside* CI/CD, you'd swap GitHub's free runner for a
  self-hosted GPU runner or a paid Colab Pro + API setup -- ask me if you get there.

## Project structure

```
ml-cicd-project/
├── .github/workflows/mlops-pipeline.yml   # the CI/CD pipeline
├── src/
│   ├── preprocess.py    # shared data loading/splitting (fixed random_state)
│   ├── train.py         # trains a candidate model
│   └── evaluate.py      # scores candidate, promotes if it beats production
├── serving/
│   ├── app.py           # FastAPI inference server
│   └── Dockerfile        # optional container-based deploy path
├── tests/test_model.py
├── models/               # candidate_model.pkl, production_model.pkl, metrics (generated)
└── requirements.txt
```

## 1. Local setup (to test before pushing)

```bash
cd ml-cicd-project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd src
python train.py       # creates models/candidate_model.pkl + candidate_metrics.json
python evaluate.py     # scores it, promotes to production if it's better
```

First run always promotes (there's no production model yet to beat).

Test the API locally:
```bash
cd ../serving
uvicorn app:app --reload --port 8000
# then POST to http://localhost:8000/predict with a JSON body matching PredictionRequest
```

## 2. Using Google Colab for experimentation

1. Push this project to a GitHub repo.
2. In Colab: `Runtime > Change runtime type > GPU` (if your model needs it).
3. In a Colab cell:
   ```python
   !git clone https://github.com/<you>/<your-repo>.git
   %cd <your-repo>
   !pip install -r requirements.txt
   %cd src
   !python train.py
   !python evaluate.py
   ```
4. Iterate on `train.py` (bigger model, more data, different algorithm, hyperparameter
   search) directly in the Colab cells until you're happy with the accuracy.
5. Copy your final changes back into `src/train.py` in the repo (edit the file directly on
   GitHub, or from Colab push back with a token):
   ```python
   !git config --global user.email "you@example.com"
   !git config --global user.name "your-name"
   !git add src/train.py
   !git commit -m "Improve training script"
   !git remote set-url origin https://<GITHUB_TOKEN>@github.com/<you>/<your-repo>.git
   !git push
   ```
   Generate a token at GitHub Settings > Developer settings > Personal access tokens
   (fine-grained, `repo` scope only).

Once `train.py` is committed, the automated pipeline uses it going forward -- no need to
run anything in Colab again unless you're experimenting with something new.

**This entire section is optional.** If your GitHub Actions pipeline is already training,
evaluating, promoting, and deploying successfully (check the Actions tab), you do not need
to touch Colab at all -- it exists purely as a tool for experimenting with bigger models or
GPU-accelerated training before committing a final `train.py`.

## 3. Deployment target: Render (this is what's actually set up)

This project deploys to **Render.com's free web service tier** -- no VPS, no SSH, no
systemd. Render is connected directly to this GitHub repo and auto-deploys whenever `main`
changes.

Setup (already done if you followed along, documented here for reference / rebuilding):
1. render.com > **New > Web Service** > connect this GitHub repo.
2. Root Directory: blank (repo root). Build Command: `pip install -r requirements.txt`.
   Start Command: `uvicorn serving.app:app --host 0.0.0.0 --port $PORT`. Instance: Free.
3. A `.python-version` file (`3.11.10`) is committed at the repo root -- Render reads this
   to avoid defaulting to an unsupported Python version.
4. Auto-Deploy must be set to **On Commit** in the service's Settings > Build & Deploy
   (not "After CI Checks Pass" -- promotion commits from the pipeline don't carry a CI
   check of their own, so that mode would silently never deploy).
5. If deploys don't trigger on push, check that Render's GitHub App has access to this
   specific repo at github.com/apps/render/installations/new.

No GitHub secrets are needed for deployment itself -- Render watches the repo on its own.
The optional `RENDER_DEPLOY_HOOK_URL` secret (used in the workflow's last step) just makes
the redeploy fire a few seconds faster; everything works without it too.

Free tier note: the service spins down after ~15 min idle; the first request after that
takes 30-50 seconds to wake back up. That's expected, not a bug.

## 4. GitHub repo secrets

None are required for the current Render-based setup. The model-promotion commit step
uses the built-in `GITHUB_TOKEN` (available automatically to every workflow run), enabled
by `permissions: contents: write` in the workflow file.

The only optional secret:

| Secret name              | Value                                              |
|---------------------------|-----------------------------------------------------|
| `RENDER_DEPLOY_HOOK_URL`  | from Render service > Settings > Deploy Hook (optional, speeds up redeploys) |

## 5. Run it

- **Automatically**: push a change under `src/`, or wait for the weekly schedule (edit the
  `cron` line in `mlops-pipeline.yml` to change frequency).
- **Manually**: GitHub repo > Actions tab > "MLOps CI/CD Pipeline" > Run workflow.

Check the run logs -- `evaluate.py`'s prints show the candidate accuracy, the current
production accuracy, and whether it promoted/deployed.

## 6. On accuracy / "enough data"

The included dataset (UCI Adult Income, ~48K rows) is there so the pipeline has something
real to run against out of the box. For your own accuracy needs:
- Swap `DATA_URL`/`COLUMNS`/feature lists in `src/preprocess.py` for your dataset.
- More data generally helps, but the bigger accuracy levers are: feature quality, class
  balance, and model/hyperparameter choice (in `train.py`) -- try Colab with GPU for faster
  iteration on those before committing the final script.
- The promotion gate in `evaluate.py` already guarantees the pipeline never regresses --
  every deploy is measured on the same held-out data as every model before it.

## 7. Real-time drift and bias monitoring

This is the second half of the project: a scheduled job that watches live traffic on the
deployed API, checks it against the training-time data distribution, checks fairness across
sensitive groups, and automatically re-triggers the retraining pipeline if something looks
wrong. It's the more research-relevant half -- see the docstring in `monitoring/drift_check.py`
for the methodology and citations (Population Stability Index; the four-fifths fairness rule).

### How it fits together

- `serving/app.py` logs every `/predict` call in memory (features + output + timestamp),
  exposed at `GET /monitoring/raw-data` (API-key protected).
- `monitoring/drift_check.py` (run on a schedule by GitHub Actions) fetches that log,
  computes PSI per feature against `models/reference_stats.json` (built during training),
  checks the four-fifths rule across `sex` and `race`, saves a timestamped report under
  `monitoring/history/`, POSTs the latest report to the API, and -- if either check fails --
  calls the GitHub API to re-run `mlops-pipeline.yml`.
- `GET /dashboard` on your deployed API renders the latest report as a live page.

**Important limitation to know and be able to explain**: the prediction log lives in memory
on the Render instance, so it resets whenever the service restarts, redeploys, or spins down
from inactivity (free tier). This is fine for demonstrating the mechanism end-to-end, but a
production system would persist this to a real database. Worth stating as a known limitation
if you discuss this project, not something to hide.

### Setup

1. **Set an API key** so your monitoring endpoints aren't public to anyone who finds the URL:
   - Pick any random string, e.g. generate one with:
     ```powershell
     python -c "import secrets; print(secrets.token_hex(16))"
     ```
   - Render dashboard > your service > **Environment** > add `MONITOR_API_KEY` = that value.
   - GitHub repo > **Settings > Secrets and variables > Actions** > add secret
     `MONITOR_API_KEY` = the same value.

2. **Tell the monitor where your API lives**:
   - GitHub repo secrets > add `API_BASE_URL` = `https://ml-income-api.onrender.com`
     (your actual Render URL, no trailing slash).

3. Push everything (`monitoring/`, updated `.github/workflows/`, updated `app.py`,
   `train.py`, `evaluate.py`) to GitHub, and let the existing `mlops-pipeline.yml` run once
   (manually trigger it from the Actions tab) so `models/reference_stats.json` gets created
   and committed -- the drift checker needs this file to exist in the repo.

### Generate demo traffic (do this to actually see it work)

A brand-new deploy has zero logged predictions, so there's nothing to analyze yet. From your
machine:

```powershell
cd C:\Users\Intel\Downloads\ml-cicd-project\ml-cicd-project
pip install requests
python monitoring/simulate_traffic.py --url https://ml-income-api.onrender.com --mode normal --n 100
```

Then run the drift check once manually to confirm it reports "OK":
```powershell
$env:API_BASE_URL = "https://ml-income-api.onrender.com"
$env:MONITOR_API_KEY = "<the key you set above>"
python monitoring/drift_check.py
```

Now prove it actually catches a real shift:
```powershell
python monitoring/simulate_traffic.py --url https://ml-income-api.onrender.com --mode shifted --n 150
python monitoring/drift_check.py
```
You should see one or more features with PSI above 0.25 and `"alert": true` in the printed
report, and (if `GITHUB_TOKEN`/`GITHUB_REPOSITORY` are set -- true automatically when this
runs inside GitHub Actions) the retraining pipeline gets triggered automatically.

### View the live dashboard

```
https://ml-income-api.onrender.com/dashboard
```

### Turn on the schedule

`drift-monitor.yml` already runs every 6 hours (`cron: "0 */6 * * *"`) with no further setup
needed once the two secrets above are in place. Adjust the cron expression to match your
expected traffic volume -- checking every 6 hours is arbitrary and easy to change.

### For your research write-up

Worth explicitly stating if you present this project: PSI and the four-fifths rule are
simple, interpretable screens, not a complete fairness or drift audit. Natural extensions to
mention as future work: calibration/equalized-odds fairness metrics, a proper drift test
(e.g. Kolmogorov-Smirnov per feature with multiple-testing correction), and persistent
storage for the prediction log so drift can be tracked over long time windows rather than
just "since the last restart."
