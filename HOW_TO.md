### 1. Build binary wheel and install locally
```shell
python3 setup.py bdist_wheel
pip3 install -e .
# pip install dscollection
```

### 2. Upload to TestPyPI
```shell
pip3 install --upgrade twine

# generate distribution archives 
python3 -m build

# upload distributions to testpypi
twine upload --repository testpypi dist/*
```

### 3. Install newly uploaded package using pip
```shell
pip install -i https://test.pypi.org/simple/ dscollection 
```