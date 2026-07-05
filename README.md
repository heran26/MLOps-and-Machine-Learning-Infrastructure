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

## 3. Set up the cloud server (where the model gets deployed to)

Any small VPS works (DigitalOcean, Linode, a free-tier AWS/GCP VM, etc.). On the server:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
mkdir -p ~/ml-app/models
cd ~/ml-app
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn scikit-learn pandas joblib pydantic
```

Create a systemd service so the API survives reboots/crashes -- `/etc/systemd/system/ml-app.service`:

```ini
[Unit]
Description=ML Inference API
After=network.target

[Service]
User=<your-ssh-user>
WorkingDirectory=/home/<your-ssh-user>/ml-app
ExecStart=/home/<your-ssh-user>/ml-app/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ml-app.service
sudo systemctl start ml-app.service
```

Do one manual first deploy so `models/production_model.pkl` and `serving/app.py` exist on
the server before the pipeline tries to restart the service.

## 4. GitHub repo secrets

In your GitHub repo: **Settings > Secrets and variables > Actions > New repository secret**

| Secret name        | Value                                                    |
|---------------------|-----------------------------------------------------------|
| `SSH_HOST`          | server IP or hostname                                    |
| `SSH_USER`          | SSH username on the server                                |
| `SSH_PRIVATE_KEY`   | private key that matches a public key in the server's `~/.ssh/authorized_keys` |

No secret is needed for the git commit-back step -- the built-in `GITHUB_TOKEN` (available
automatically to every workflow) handles that, since `permissions: contents: write` is set
in the workflow file.

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

## 7. Extending to monitoring

Natural next additions once this is running:
- Log every `/predict` request/response in `app.py` to a file or database.
- A scheduled job comparing recent live prediction distributions to training-data
  distributions (data drift).
- Alerting (Slack/email) on the workflow's success/failure via GitHub Actions notifications.

Ask if you want any of these built out next.
