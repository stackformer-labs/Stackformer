from setuptools import setup, find_packages

setup(
    name="Stackformer",
    version="0.1.2",
    description="Modular transformer blocks built in PyTorch",
    # long_description=open("README.md", "r", encoding="utf-8").read(),
    # long_description_content_type="text/markdown",
    author="Gurumurthy",
    author_email="gurumurthy.00300@gmail.com",
    url="https://github.com/Gurumurthy30/Stackformer",
    project_urls={
        "Repository": "https://github.com/Gurumurthy30/Stackformer",
        "Issue Tracker": "https://github.com/Gurumurthy30/Stackformer/issues",
        "Discussions": "https://github.com/Gurumurthy30/Stackformer/discussions",
    },
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        "torch>=2.0,<2.6",
        "tqdm>=4.67",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)