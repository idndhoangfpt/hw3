# Notebook Workflow

Use notebooks in `notebooks/` to train or retrain models.

Rules:

- New notebook runs save outputs to `notebook_outputs/`
- The Streamlit app still reads the current demo models from `artifacts/`
- Testing a notebook will not overwrite the current app models

Output folders:

- `notebook_outputs/models/`
- `notebook_outputs/reports/`
- `notebook_outputs/predictions/`

Suggested flow:

1. Open a notebook in `notebooks/`
2. Train or retrain there
3. Check the generated files in `notebook_outputs/`
4. Only copy into `artifacts/` if you explicitly want the app to use the new model
