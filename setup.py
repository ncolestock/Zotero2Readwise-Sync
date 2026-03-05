from setuptools import setup, find_packages

setup(
    name="zotero2readwise-sync",
    version="0.1.0",
    py_modules=["run"],
    install_requires=[
        "requests",
        "pyzotero==1.10.0",
    ],
    python_requires=">=3.7",
)
