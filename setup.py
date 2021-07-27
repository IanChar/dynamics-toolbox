import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dynamics_toolbox", # Replace with your own username
    version="1.0.0",
    author="Ian Char",
    author_email="ichar@cs.cmu.edu",
    description=("A library for easily training and using dynamics models."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IanChar/dynamics-toolbox",
    packages=setuptools.find_packages(),
    # install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
