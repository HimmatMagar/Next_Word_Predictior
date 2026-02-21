import setuptools


with open("README.md", 'r', encoding='UTF-8') as f:
      long_description = f.read()


__version__ = "0.0.0.0"


REPO_NAME = "Next_Word_Predictior"
AUTHOR_NAME = "HimmatMagar"
SRC_REPO = "nextWordPrediction"
AUTHOR_EMAIL = "himmatmagar007@gmail.com"

setuptools.setup(
      name="Next Word Prediction",
      version=__version__,
      author=AUTHOR_NAME,
      author_email=AUTHOR_EMAIL,
      description="End to End DL implementation for Next Word Prediction",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
      project_urls={
            "Bug Tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issue"
      },
      package_dir={"": "src"},
      packages=setuptools.find_packages(where='src')
)
