from setuptools import find_packages, setup


setup(
    name="diffusion-planner-from-scratch",
    version="0.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    extras_require={"dev": ["pytest>=8.0", "ruff>=0.9", "nbformat>=5.10"]},
)
