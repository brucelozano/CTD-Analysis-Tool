from setuptools import find_packages, setup


setup(
    name="ctd-tool",
    version="0.1.0",
    description="CNV-first CTD analysis tool for profile parsing, QC, and plotting.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "xarray>=2024.1.0",
        "pydap>=3.4.1",
    ],
    extras_require={
        "dev": ["pytest>=7.4"],
    },
    entry_points={"console_scripts": ["ctd-tool=ctd_tool.cli:main"]},
    python_requires=">=3.9",
)
