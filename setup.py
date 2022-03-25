import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.rstrip() for line in fh.readlines()]

setuptools.setup(
    name="speedtoxify",
    version="0.0.2",
    author="Andy Lo",
    author_email="andylolu24@gmail.com",
    url="https://github.com/andylolu2/speedtoxify",
    description="Wrapper around detoxify package for faster inference using ONNX runtime.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        include=["speedtoxify"], exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
