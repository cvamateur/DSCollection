from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    # name what to pip install, different from module name to import from code
    name="dscollection",
    author="Chris LIU",
    author_email="chris.lq@hotmail.com",
    version="1.0.0",  # 0.0.x is imply it unstable
    description="A collection of tools ease of dataset manipulation",
    scripts=["bin/dsc"],
    packages=find_packages("src"),  # a list of actual python modules
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "tqdm ~= 4.64.0",
        "pycocotools ~= 2.0.4",
        "pandas ~= 1.4.3",
        "numpy >= 1.17.0",
        "matplotlib ~= 3.5.1",
    ]
)
