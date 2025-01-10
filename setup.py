# setup.py
from setuptools import setup, find_packages

setup(
    name="ai_developer_worker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.9.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.2",
        "httpx>=0.24.1",
        "pytest>=8.0.0",
        "pytest-asyncio>=0.23.0"
    ],
    entry_points={
        'console_scripts': [
            'ai-dev=src.main:main',
        ],
    },
)