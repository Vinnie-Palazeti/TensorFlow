import setuptools

setuptools.setup(
    name="bootstraps",  # Replace with your own username
    version="0.0.1",
    author="Vinnie Palazeti",
    author_email="vinnie.palazeti@gmail.com",
    description="looking into bootstrap + NNs",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'tensorflow',
        'numpy',
        'jupyter'
    ],
    python_requires='>=3.6',
)