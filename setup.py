from setuptools import setup

setup(
    name="hyperliquidmaster",
    version="0.1.0",
    package_dir={"": "src"},
    packages=["hyperliquidmaster"],
    install_requires=[
        "hyperliquid-python-sdk>=0.15.0",
        "pandas>=2.2.0",
        "numpy>=2.2.0",
        "requests>=2.32.0",
        "web3>=7.12.0",
        "eth-account>=0.13.0",
        "aiohttp>=3.7.4",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.6.0",
        "ta>=0.11.0",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "schedule>=1.1.0",
        "colorlog>=6.6.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "hyperliquid=hyperliquidmaster.scripts.master_bot:main",
        ],
    },
)
