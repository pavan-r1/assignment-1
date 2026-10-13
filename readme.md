# Customer Churn Prediction MLOps

An end-to-end Machine Learning Operations (MLOps) project that predicts customer churn using machine learning and demonstrates model training, experiment tracking, API development, Docker containerization, automated testing, and CI/CD with GitHub Actions.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [MLOps Workflow](#mlops-workflow)
- [Installation](#installation)
- [Model Training](#model-training)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Run the API Locally](#run-the-api-locally)
- [API Endpoints](#api-endpoints)
- [Docker Setup](#docker-setup)
- [Automated Testing](#automated-testing)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Predicting churn helps organizations identify customers who may leave and take proactive retention measures.

This project develops a machine learning pipeline to predict customer churn based on customer-related features. The trained model is exposed through a FastAPI application, containerized with Docker, and integrated with automated development workflows.

The project demonstrates how machine learning can be combined with software engineering and MLOps practices to support reproducible training, consistent deployment, and maintainable model-serving applications.

## Objectives

- Build a machine learning model for customer churn prediction.
- Prepare and process customer data for model training.
- Evaluate model performance using suitable classification metrics.
- Track experiments and model results using MLflow.
- Serve predictions through a REST API.
- Package the application and dependencies with Docker.
- Automate validation and image builds through GitHub Actions.
- Maintain source code and configuration using Git and GitHub.

## Features

- **Data Processing:** Loads and prepares data for machine learning.
- **Model Training:** Trains a classification model to predict churn.
- **Model Evaluation:** Supports evaluation using metrics such as F1-score, precision, recall, and accuracy, depending on the implementation.
- **Experiment Tracking:** Uses MLflow to record training runs and metrics when configured.
- **REST API:** Provides an interface for submitting customer data and receiving predictions.
- **Docker Containerization:** Packages the API and its dependencies into a container image.
- **Automated CI/CD:** Uses GitHub Actions to execute configured checks and build workflows.
- **Testing:** Organizes automated tests in a dedicated directory.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning and evaluation |
| MLflow | Experiment tracking and model logging |
| FastAPI | REST API development |
| Uvicorn | ASGI application server |
| Joblib | Model serialization, if used |
| Docker | Containerization |
| Git | Version control |
| GitHub | Repository hosting |
| GitHub Actions | CI/CD automation |
| Pytest | Automated testing, if configured |

The exact libraries and versions are defined by the project's dependency files.

## Project Structure

```text
assignment1/
├── .github/
│   └── workflows/
│       └── docker-image.yml
├── data/
│   └── ...
├── models/
│   └── ...
├── src/
│   ├── train.py
│   ├── api.py
│   └── ...
├── tests/
│   └── ...
├── .dockerignore
├── .gitignore
├── Dockerfile
├── readme.md
├── requirements.txt
└── requirements-freeze.txt
```

**Directory descriptions:**

- `data/`: Contains the dataset used for training and evaluation.
- `models/`: Stores trained model artifacts when the project is configured to save them here.
- `src/`: Contains model-training scripts, API code, and supporting modules.
- `tests/`: Contains automated tests.
- `.github/workflows/`: Stores GitHub Actions workflow files.
- `Dockerfile`: Defines the Docker image build process.
- `.dockerignore`: Excludes unnecessary files from the Docker build context.
- `.gitignore`: Excludes files that should not be tracked by Git.
- `requirements.txt`: Lists the core Python dependencies.
- `requirements-freeze.txt`: Records a larger set of installed package versions for environment reproducibility.
- `readme.md`: Documents the project and its usage.

The filenames shown inside the directories are illustrative; the actual repository may use different names.

## MLOps Workflow

The project follows this general workflow:

```text
Customer Dataset
       |
       v
Data Preparation
       |
       v
Model Training
       |
       v
Model Evaluation
       |
       v
MLflow Experiment Tracking
       |
       v
Saved Model Artifact
       |
       v
FastAPI Prediction Service
       |
       v
Docker Image
       |
       v
GitHub Actions CI/CD
       |
       v
Deployment Environment
```

The exact order and automation depend on the implemented training script and GitHub Actions workflow.

## Installation

### Prerequisites

Install the following tools:

- Python 3.12, if compatible with the project's dependencies.
- Git.
- Docker Desktop or Docker Engine.
- A GitHub account.

### 1. Open the project directory

```powershell
cd "C:\Users\Pavan R\OneDrive\Documents\MLops\assignment1"
```

### 2. Activate the virtual environment

If the shared virtual environment is located at `C:\Users\Pavan R\OneDrive\Documents\MLops\.venv`, activate it using:

```powershell
& "C:\Users\Pavan R\OneDrive\Documents\MLops\.venv\Scripts\Activate.ps1"
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project needs additional dependencies for training, testing, or MLflow, ensure those packages are included in the relevant dependency file.

## Model Training

If the training script is `src/train.py`, run:

```powershell
python src/train.py
```

The training process may include:

1. Loading the customer dataset.
2. Preparing input features and the target variable.
3. Preprocessing data.
4. Splitting the dataset into training and validation sets.
5. Training a classification model.
6. Evaluating the model.
7. Saving the trained model.
8. Logging metrics and artifacts to MLflow, if configured.

### Model Evaluation

Useful classification metrics include:

- **Accuracy:** Proportion of predictions that are correct.
- **Precision:** Proportion of predicted churn cases that are actual churn cases.
- **Recall:** Proportion of actual churn cases identified by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Summarizes correct and incorrect classifications.

For churn prediction, recall and F1-score can be particularly useful because missing a customer who is likely to leave may have business consequences.

The actual evaluation metrics must be obtained by running the training pipeline. They should not be inferred from the documentation.

## MLflow Experiment Tracking

MLflow can be used to track training experiments, parameters, metrics, and model artifacts.

If the project logs experiments using MLflow, start the local tracking interface with:

```powershell
mlflow ui
```

Open:

http://127.0.0.1:5000

If a tracking database or artifact directory is configured, ensure that the MLflow UI uses the appropriate tracking URI and that the relevant artifacts are accessible.

Experiment tracking features depend on the implementation of the training script.

## Run the API Locally

If the API is implemented in `src/api.py`, and the FastAPI application object is named `app`, start the server with:

```powershell
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Open the interactive documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

**Important:** These commands assume the module is `src/api.py` and the application object is named `app`. Adjust the command if your actual entry point differs.

## API Endpoints

A typical churn prediction API may expose the following endpoints. The exact endpoints depend on the implementation.

| HTTP method | Example path | Purpose |
|---|---|---|
| GET | `/` | Application information |
| GET | `/health` | Health check |
| POST | `/predict` | Predict customer churn |

### Example prediction request

The following payload is illustrative only. Use the fields and data types required by your API's request schema.

```json
{
  "tenure": 12,
  "monthly_charges": 70.5,
  "total_charges": 846.0
}
```

### Example prediction response

```json
{
  "prediction": 1
}
```

In this example, `1` represents the churn class and `0` represents the non-churn class, assuming that mapping is used by the trained model.

The actual feature names, required fields, and response format must match the implemented API.

## Docker Setup

Docker packages the application and its dependencies into a container.

### 1. Build the Docker image

Run this command from the directory containing the Dockerfile:

```powershell
docker build -t churn-assignment1-api:latest .
```

### 2. Run the container

```powershell
docker run --rm -p 8000:8000 churn-assignment1-api:latest
```

### 3. Access the API

If the application starts successfully, open:

http://localhost:8000/docs

### 4. Check running containers

```powershell
docker ps
```

### 5. Stop a container

If running the container in detached mode, use its container name or ID:

```powershell
docker stop <container_name_or_id>
```

The Dockerfile must install the required packages, copy the application files, include or otherwise provide the trained model, and start the API using the correct module path.

## Automated Testing

The `tests/` directory is intended to contain automated tests for the project.

If Pytest is installed and the tests are configured, run:

```powershell
pytest -v
```

Recommended test areas include:

- Dataset loading and preprocessing.
- Model training and prediction.
- Model artifact loading.
- API input validation.
- Prediction response structure.
- Health-check endpoint.
- Docker image build and application startup.

Only claim that a test passed after verifying its execution result.

## GitHub Actions CI/CD

GitHub Actions automates selected development and deployment tasks.

The workflow files are stored in:

```text
.github/workflows/
```

A typical Docker workflow can:

1. Check out the repository.
2. Set up Docker Buildx.
3. Build the Docker image.
4. Run tests or validation steps, if configured.
5. Push the image to a container registry, if registry credentials and push settings are configured.

### Verify workflow execution

1. Open the GitHub repository.
2. Click the **Actions** tab.
3. Select the latest workflow run.
4. Open the job logs.
5. Identify any failed step and correct the issue.
6. Commit and push the fix to trigger another run.

A successful image build does not necessarily mean the image has been pushed or the API deployed. Those steps require additional configuration.

## Deployment

The API can be deployed to a suitable hosting platform that supports Python applications or Docker containers.

Before deployment, verify that:

- The API starts successfully.
- All dependencies are installed.
- The trained model is available.
- The application binds to the host and port required by the platform.
- Required environment variables are configured.
- The health-check endpoint, if implemented, works.
- Secrets are stored securely.

The actual deployment URL should be added here once the deployment is verified.

## Troubleshooting

### 1. `ModuleNotFoundError`

**Cause:** A required Python package is missing.

**Solution:**

```powershell
pip install -r requirements.txt
```

If the error persists, identify the missing package and add a compatible version to the appropriate dependency file.

### 2. Docker build failure

**Cause:** The Dockerfile may be empty, contain invalid instructions, or reference files that do not exist.

**Solution:**

```powershell
Get-Content Dockerfile
docker build --no-cache -t churn-assignment1-api:latest .
```

Check that the Dockerfile is valid and that all referenced files exist.

### 3. API import error

**Cause:** The Uvicorn module path may not match the location of the API file or the FastAPI application variable.

**Solution:** Verify the entry-point file and application variable before starting Uvicorn.

### 4. Model file not found

**Cause:** The API expects a model artifact at a path that does not exist in the current environment or container.

**Solution:** Run the training pipeline, verify the output path, and ensure the artifact is available to the API.

### 5. GitHub Actions dependency error

**Cause:** A package used in the workflow or application is missing from the dependency file.

**Solution:** Add the required dependency to `requirements.txt`, commit the change, and push it to GitHub.

### 6. MLflow artifact or model-loading error

**Cause:** The saved model may reference incompatible package versions or an artifact path that is not available.

**Solution:** Keep training and serving dependencies compatible, verify the artifact location, and test loading the model in a clean environment.

## Future Improvements

- Add stronger input validation and data-quality checks.
- Compare multiple classification algorithms.
- Improve model evaluation with cross-validation.
- Track model versions and experiments consistently.
- Add automated unit and integration tests.
- Introduce model and data drift monitoring.
- Automate model retraining and evaluation.
- Publish versioned Docker images to a container registry.
- Add secure cloud deployment and monitoring.
- Document the final verified evaluation results.

## Author

**Pavan R.**

B.E. in Data Science  
T. John Institute of Technology, Bengaluru

GitHub: https://github.com/pavan-r1

## License

Add an appropriate license if you intend to distribute this project as open-source software. Until a license is included, do not assume that others have permission to reuse or redistribute the project.

---

*This project is intended for educational and demonstration purposes. Predictions are estimates produced by a machine learning model and should be evaluated before being used for business decisions.*
