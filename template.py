import os

# Project structure
project_structure = {
    "README.md": "",
    "requirements.txt": "",
    "data": ["patient_data.csv"],
    "notebooks": ["eda_and_model.ipynb"],
    "scripts": [
        "data_preprocessing.py",
        "model_training.py",
        "prediction.py",
        "visualization.py"
    ],
    "outputs": {
        "trained_model.pkl": "",
        "predictions.csv": "",
        "figures": []
    }
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        elif isinstance(content, list):
            os.makedirs(path, exist_ok=True)
            for file in content:
                file_path = os.path.join(path, file)
                with open(file_path, 'w') as f:
                    pass
        else:
            with open(path, 'w') as f:
                f.write(content)

if __name__ == "__main__":
    base_path = os.getcwd()  # Current directory
    create_structure(base_path, project_structure)
    print("Project folder structure and files created successfully!")