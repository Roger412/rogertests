from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np

# Example:
# X: shape (n_trials, n_channels, n_samples)
# y: shape (n_trials,), values 0 (left) or 1 (right)

# Create pipeline
csp = CSP(n_components=4, reg=None, log=True, transform_into='csp_space')
lda = LinearDiscriminantAnalysis()
clf = make_pipeline(csp, lda)

# Evaluate with cross-validation (optional but useful)
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validated accuracy: %f ± %f" % (scores.mean(), scores.std()))

# Fit model to all data
clf.fit(X, y)
