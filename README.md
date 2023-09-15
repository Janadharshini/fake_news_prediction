# FAKE NEWS PREDICTION


Import necessary libraries:
I import various libraries required for data processing, model training, and evaluation. Some of the important ones include sklearn for machine learning, pandas for data manipulation, and matplotlib for visualization.


Read the dataset:

I read a dataset from a CSV file named 'fake_or_real_news.csv' using pd.read_csv and store it in a DataFrame called 'news'.


Split the dataset:

I split the dataset into training and testing sets using the train_test_split function from sklearn.model_selection. The training set contains 80% of the data, while the testing set contains 20%.


Text Vectorization:

I use TfidfVectorizer from sklearn.feature_extraction.text to convert the text data (news articles) into TF-IDF (Term Frequency-Inverse Document Frequency) vectors. This is a common technique for representing text data in a format that machine learning models can understand.


Model Training:

I train six different machine learning models: Passive Aggressive Classifier, Gaussian Naive Bayes, Decision Tree Classifier, Random Forest Classifier, Support Vector Classifier (SVC), and Logistic Regression. Each model is trained on the TF-IDF transformed training data.


Model Prediction:

I use the trained models to make predictions on the test data.


Model Evaluation:

I calculate the accuracy of each model using accuracy_score from sklearn.metrics. This gives you an idea of how well each model performs on the test data.


Data Visualization:

I create histograms to visualize the distribution of the length of news titles for real and fake news articles.
I create a bar chart to compare the accuracy of the different machine learning models.


Confusion Matrices:
You calculate and display confusion matrices for each model using confusion_matrix from sklearn.metrics and visualize them using plot_confusion_matrix from mlxtend.plotting. Confusion matrices help you understand the model's performance in terms of true positives, true negatives, false positives, and false negatives.
Overall, this code provides a comprehensive analysis of different machine learning models for classifying news articles as real or fake and visualizes the results for better understanding and comparison.





