# GitHub repo classification using KNN


### Requirements

Only python (3) is required. All packages specified in requirements.txt (except jupyter to read notebooks)


The bulk of the code is provided in knn.ipynb. It is in a notebook format to facilitate running different parts of the code in isolation. The main idea behind this project is described as follows:

- Start with a large dataset of unclassified repositories
- Classify a small subset of the respositories by hand (which is now the "ground truth" subset)
- Iteratively expand the subset using the following method:
	- Classify the rest of the repositories using the current ground truth subset
	- Retain the high-confidence classifications and add them to the ground truth subset (inspect by hand and do some manual corrections)
	- Repeat until as many repos are well classified as possible

Classification is done using KNN in our case. The READMEs of the repositories are vectorised using TF-IDF, then SVD is used for dimensionality reduction. The manual inspection part is quite important, as otherwise not much real "information" is gained from new classifications (the decision boundary doesn't move much).

## Note:
The knn.ipynb file is not meant to be run from start to finish all at once. It contains some example code and helper code that should not run every time.
