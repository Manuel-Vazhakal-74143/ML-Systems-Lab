import sys
import importlib.metadata

def main():
    print(f"Python version: {sys.version.split()[0]}\n")
    
    # List of packages from requirements.txt
    packages = [
        "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", 
        "seaborn", "torch", "torchvision", "torchinfo", "thop", 
        "onnx", "onnxruntime", "mlflow", "memory-profiler", "psutil", 
        "codecarbon", "fastapi", "uvicorn", "pytest", "httpx", 
        "locust", "requests", "pyarrow", "joblib", "tqdm"
    ]
    
    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg)
            print(f"{pkg}=={version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg} is NOT installed")

if __name__ == "__main__":
    main()