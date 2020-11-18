import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="differtless",
    version="0.0.1",
    author="Anastasia Ershova, Mark Penrod, Will Wang, Teresa Datta",
    author_email="aershova@g.harvard.edu, mpenrod@g.harvard.edu, willwang@g.harvard.edu, tdatta@g.harvard.edu",
    description="Package for (effortless) automatic differentiation in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/differtless/cs107-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)