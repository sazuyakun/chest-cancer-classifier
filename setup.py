import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version__ = "0.0.0"

REPO_NAME = "chest-cancer-classifier"
AUTHOR_USER_NAME = "sazuyakun"
SRC_REPO = "chestCancerClassifier"
AUTHOR_EMAIL = "sohamsamal37@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A chest cancer classification project using MLOps practices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "Flask==3.1.1",
        "PyYAML==6.0.2",
        "dvc==3.60.1",
        "mlflow==3.1.0",
        "numpy==2.3.1",
        "pandas==2.3.0",
        "torch==2.7.1",
        "matplotlib==3.10.3",
        "seaborn==0.13.2",
        "joblib==1.5.1",
        "tqdm==4.67.1",
        "python-box==7.3.2",
        "scipy==1.16.0",
        "gdown==5.2.0",
        "flask-cors==6.0.1",
        "notebook==7.4.3",
    ],
)
