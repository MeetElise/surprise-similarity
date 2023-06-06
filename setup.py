from pathlib import Path
from setuptools import find_packages, setup

README_txt = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

requirements = [
    "datasets>=2.3.2",
    "scikit-learn>=1.0",
    "english_words==1.1.0",
    "sentence_transformers>=2.2.2",
    "pandas>=1.4.1",
]

setup(
    name="surprise_similarity",
    version="0.0.4",
    description="Context-aware similarity score for embedding vectors",
    long_description=README_txt,
    long_description_content_type="text/markdown",
    author="Thomas C. Bachlechner, Mario Martone, Marjorie Schillo",
    author_email="thomas@eliseai.com",
    url="https://github.com/MeetElise/surprise-similarity",
    download_url="https://github.com/MeetElise/surprise-similarity",
    license="Apache 2.0",
    packages=["surprise_similarity"],
    install_requires=requirements,
    keywords="nlp, machine learning, fewshot learning, transformers",
    zip_safe=False,
)
