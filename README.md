# Getting Started with Homework project

## Project structure
```
MLOps_course_AGH_LAB01_HOMEWORK/ 
│ 
├── tests/
│   ├── __init__.py
│   ├── test_inference.py
│   ├── test_input.py
│   ├── test_model.py
│   └── test_response.py
│
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── app.py
├── docker-compose.yaml
├── Dockerfile
├── main.py
├── pyproject.toml
├── README.md
├── test_app.http
└── uv.lock.md
```
## Models preparation

To run the application, you need to place the pre-trained models in the correct directory.
1. Create a models folder in the root of the project (if it does not already exist)
2. Download models from: https://drive.google.com/file/d/1NRZdYq5jweVRUzAZG518LMhs4E56IgxG/view?usp=share_link
3. Copy your existing model files into the models directory.
```
MLOps_course_AGH_LAB01_HOMEWORK/ 
│
├── models/
│   ├── sentence_transformer.model
│   └── classifier.joblib
```
