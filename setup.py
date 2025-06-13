from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    # Remove diffvg from requirements as it needs manual installation
    requirements = [req for req in requirements if not req.startswith("git+")]

setup(
    name="triangulate-ai",
    version="1.0.0",
    author="TriangulateAI Team",
    description="Neural network for image-to-triangle stylization using differentiable rendering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/triangulate-ai",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "triangulate_ai=triangulate_ai.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "triangulate_ai": ["*.yaml"],
    },
)