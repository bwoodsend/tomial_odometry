from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).resolve().parent

readme = (HERE / 'README.rst').read_text("utf-8")

setup(
    author="Brénainn Woodsend",
    author_email='bwoodsend@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description=
    "Find and normalise the position and orientation of a digital dental model.",
    install_requires=[
        'numpy',
        'motmot >= 0.1.0',
    ],
    extras_require={
        "test": [
            'pytest>=3', 'pytest-order', 'coverage', 'pytest-cov',
            'tomial_tooth_collection_api @ git+ssh://git@github.com/bwoodsend/tomial_tooth_collection_api.git@9c50e74a209443992f8e54dbf29a450dbbbae3ae'
        ]
    },
    license="MIT license",
    long_description=readme,
    package_data={"tomial_odometry": []},
    keywords='tomial_odometry',
    name='tomial_odometry',
    packages=find_packages(include=['tomial_odometry', 'tomial_odometry.*']),
    url='https://github.com/bwoodsend/tomial_odometry',
    version="0.1.0",
    zip_safe=False,
)
