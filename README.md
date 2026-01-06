Fake News Detection with Python
A machine learning project that classifies news articles as REAL or FAKE using Natural Language Processing (NLP) techniques and the PassiveAggressive Classifier algorithm.
Overview
This project addresses the growing problem of misinformation spread through social media and online platforms. Using text classification techniques, the model analyzes news articles and predicts their authenticity with high accuracy.
Key Results:

Achieved 92.82% accuracy on test data
589 true positives and 587 true negatives
42 false positives and 49 false negatives

Dataset
The project uses a political news dataset (news.csv) with the following characteristics:

Shape: 7,796 rows × 4 columns
Size: 29.2 MB
Columns:

Column 1: News ID
Column 2: Title
Column 3: Text (article content)
Column 4: Label (REAL or FAKE)



Technologies Used
Python Libraries

NumPy: Numerical computing
Pandas: Data manipulation and analysis
scikit-learn: Machine learning algorithms and tools

TfidfVectorizer: Feature extraction
PassiveAggressive Classifier: Classification model
train_test_split: Dataset splitting
accuracy_score & confusion_matrix: Model evaluation


itertools: Built-in Python library for efficient iteration

Machine Learning Approach
Feature Extraction
The project uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical features. This technique weighs word importance based on frequency within documents relative to the entire corpus.
Classification Algorithm
PassiveAggressive Classifier is employed for binary classification. This online learning algorithm:

Remains passive when predictions are correct
Updates aggressively when encountering misclassifications
Works well for large-scale text classification tasks

Model Performance
The confusion matrix shows:

True Positives (FAKE correctly identified): 589
True Negatives (REAL correctly identified): 587
False Positives (REAL incorrectly labeled as FAKE): 42
False Negatives (FAKE incorrectly labeled as REAL): 49

Overall Accuracy: 92.82%
Real-World Applications

Social media platforms filtering misinformation
News aggregators verifying article authenticity
Fact-checking organizations automating initial screening
Media literacy education tools
Browser extensions for real-time fake news detection

Future Improvements
Potential enhancements to increase accuracy and performance:

Incorporate additional feature selection methods (POS tagging, Word2Vec)
Expand training dataset size
Experiment with deep learning models (LSTM, BERT)
Add topic modeling techniques
Include source credibility scoring
Implement real-time prediction API

References

DataFlair Team. "Advanced Python Project - Detecting Fake News with Python." DataFlair, August 6, 2020. https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/
Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research 12 (2011): 2825-2830.
