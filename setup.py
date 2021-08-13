import setuptools

about = {}
with open("dfsql/__about__.py") as fp:
    exec(fp.read(), about)


with open('requirements.txt') as req_file:
    requirements = req_file.read().splitlines()

setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    url=about['__github__'],
    download_url=about['__pypi__'],
    license=about['__license__'],
    author=about['__author__'],
    author_email=about['__email__'],
    description=about['__description__'],
    packages=setuptools.find_packages(),
    install_requires=requirements,
    extras_require=dict(
        test=['pytest>=5.4.3', 'requests >= 2.22.0', 'modin[all]>=0.10.1', 'pytest-timeout>=1.4.2'],
        modin=['modin[all]>=0.10.1']),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7"
)
