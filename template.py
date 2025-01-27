import os
from pathlib import Path

package_name="LifeExpectancyPrediction"

list_of_files = [
    "github/workflows/main.yml",
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/components/data_ingestion.py",
    f"src/{package_name}/components/data_transformation.py",
    f"src/{package_name}/components/model_trainer.py",
    f"src/{package_name}/pipelines/__init__.py",
    f"src/{package_name}/pipelines/training_pipeline.py",
    f"src/{package_name}/pipelines/predict_pipeline.py",
    f"src/{package_name}/logger.py",
    f"src/{package_name}/config.py",
    "notebooks/reaserch.ipynb",
    "notebooks/data/.gitkeep",
]

for file in list_of_files:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"File {filepath} already exists and is not empty")
        continue







