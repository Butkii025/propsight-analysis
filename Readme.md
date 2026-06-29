# PropSight — Real Estate Valuation Platform

## What changed from your original version
| Bug | File | Fix |
|---|---|---|
| Default form inputs predicted **$184M** | `app.py` | Non-user features now default to dataset **medians**, not zero |
| Neighborhood selector had **zero effect** on price | `train_model.py` | One-hot Neighborhood columns are now actually trained on and saved |
| Headline R²/MAE was computed on **training data** (optimistic) | `train_model.py` | Reports honest **held-out test** R²/MAE (93.68% / $12,695) |
| "Key Drivers" chart was **hardcoded fake numbers** | `app.py` | Pulled live from `gbr_model.feature_importances_` |

Full model artifacts were retrained with `train_model.py` and saved to `models/`.

## Project structure
```
propsight/
├── app.py                  # Streamlit app (entry point)
├── train_model.py          # Retraining script — run this if you update train.csv
├── requirements.txt
├── data/
│   └── train.csv
├── models/
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   ├── gbr_model.pkl
│   ├── model_columns.pkl
│   ├── feature_medians.pkl
│   ├── feature_importances.pkl
│   └── metrics.pkl
└── Technical_Documentation.pdf   # optional, shown as a download button
```

## Deploying to Streamlit Community Cloud

1. **Create a GitHub repo** and push everything in this folder to it (including
   `data/train.csv` and the `models/*.pkl` files — they're small enough to
   commit directly, no Git LFS needed).
   ```bash
   git init
   git add .
   git commit -m "PropSight valuation app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
   Do **not** commit anything in `.ipynb_checkpoints/` — the `.gitignore` here
   already excludes it.

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with
   your GitHub account.

3. Click **"New app"**, then:
   - **Repository**: pick the repo you just pushed
   - **Branch**: `main`
   - **Main file path**: `app.py`

4. Click **Deploy**. Streamlit Cloud installs everything in `requirements.txt`
   automatically — first deploy usually takes 2–4 minutes.

5. You'll get a public URL like `https://<your-app-name>.streamlit.app` that
   anyone can open — that's the live frontend for end users.

## Updating the model later
If you retrain (e.g. with new data), just run:
```bash
python train_model.py
```
then `git add models/ && git commit -m "retrain" && git push` — Streamlit
Cloud auto-redeploys on every push to the connected branch.

## Notes
- `test.csv` and `sample_submission.csv` from the original Kaggle workflow
  aren't needed by the app — leave them out of the repo to keep it lean.
- Neighborhood effects in the UI are intentionally modest (~5–10% swings, not
  the raw 3x gap you'd see in `train.csv` between e.g. MeadowV and NoRidge).
  That's correct: most of the raw neighborhood price gap is already explained
  by quality/size, which strongly correlate with neighborhood — the model is
  showing the residual location premium *after* controlling for those.