from setuptools import setup, find_packages

setup(
    name="ma4moccmckp",
    version="0.1.0",
    description="Multi-Agent Framework for Multi-Objective Chance-Constrained  Multiple-Choice Knapsack Problem",
    author="Xuanfeng Li",
    author_email="eddylxf@outlook.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "pymoo",
        "deap",
        "pyautogen>=0.4",
        "ray",
        "streamlit",
    ],
)