"""Setup configuration for the Math Solver package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="math-solver",
    version="1.0.0",
    author="Math Solver Team",
    author_email="team@mathsolver.com",
    description="Production-level mathematics problem solver using OpenAI's API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/math-solver",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "math-solver=math_solver.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "math_solver": ["py.typed"],
    },
    zip_safe=False,
    keywords="mathematics solver openai api education",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/math-solver/issues",
        "Source": "https://github.com/yourusername/math-solver",
        "Documentation": "https://math-solver.readthedocs.io/",
    },
)
