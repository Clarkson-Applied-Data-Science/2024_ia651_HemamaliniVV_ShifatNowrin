# **NEWS CLASSIFICATION WITH NLP MODELS**

## IA653 - NATURAL LANGUAGE PROCESSING

### TEAM MEMBERS
 - HEMAMALINI VENKATESH VANETHA
 - SHIFAT NOWRIN

## PROJECT OVERVIEW

This project focuses on the classification of news articles into predefined categories using various Natural Language Processing (NLP) models. The objective is to compare the performance of different models and evaluate their effectiveness in accurately predicting the category of a given news article.

### Key Objectives

 - To explore and implement different NLP models for text classification.
 - To preprocess the data effectively to ensure high model performance.
 - To evaluate and compare the models based on performance metrics like accuracy, precision, recall, and F1-score.

 The project leverages different types of Natural Language Processing techniques to provide insights into their strengths and limitations for the task of news classification.

### DATASET

The dataset contains news articles categorized into the following five classes:

- Business
- Education
- Entertainment
- Sports
- Technology

Data has 5 different files for each category. All these files are combined and shuffled.To more specific, we combined headline,description and content in a single variable text and then assigned categroies as:

business - 0 , education - 1, entertainment - 2, sports - 3, technology - 4

Each article consists of a textual body that describes news content. The goal is to predict the category of the article based on the text.



### DATA PREPROCESSING

- Text cleaning: Remove unnecessary characters, punctuation, and special symbols.
- Tokenization: Split the text into individual words.
- Stopword Removal: Remove frequently occurring but unimportant words
- Regex: Used Regex to format the sentence.
- Lowercasing: Convert all text to lowercase to ensure uniformity.
- Feature Extraction:
TF-IDF (Term Frequency-Inverse Document Frequency): Converts textual data into numerical vectors by measuring the importance of words relative to their frequency in the document and across the corpus.

##### Train test split

Data has been divided into training and testing where y is category and x is text.

#### Distribution of Text Lengths

![Distribution of Text Lengths](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/Distribution%20of%20text%20lengths.png)

#### Distribution of Word Counts

![Distribution of Word Counts](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/Distribution%20of%20word%20counts.png)

#### Category Distribution 

![Category Distribution](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/Category%20Distribution.png)

## WORDCLOUD FOR CATEGORIES

![TECHNOLOGY](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/WC%20technology.png)


![BUSINESS](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/WC%20business.png)


![EDUCATION](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/WC%20education.png)


![SPORTS](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/WC%20sports.png)


![ENTERTAINMENT](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/WC%20entertainment.png)


## **MODELS IMPLEMENTED**

### LOGISTIC REGRESSION

- Description: 
Logistic Regression is a traditional machine learning model that works well for text classification when combined with feature extraction techniques like TF-IDF. It is based on the principle of finding a decision boundary to separate classes by optimizing the logistic loss function.

- Advantages:
Fast and efficient for high-dimensional data.
Simple to implement and interpret.

- Process:
The TF-IDF vectorizer was used to transform the text into numerical vectors.
Logistic Regression was then trained on these vectors to predict the categories.

- Evaluation:
Generated a classification report showing precision, recall, F1-score, and accuracy for each category.
Confusion matrix visualized the misclassifications.

![Classification Report for Logistic Regression](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/CR%20for%20Logistic%20Regression.png)

![Confusion Matrix for Logistic Regression](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/CM%20for%20Logistic%20regression.png)

- Logistic Regression has 98% accuracy. From confusion matrix, we can see that mostly all categories are doing well with high accuracy.

### NAIVE BAYES

- Description: Naive Bayes is a probabilistic model that works on the principle of Bayes’ theorem. It assumes that the features (words in this case) are independent of each other given the class label.

- Advantages:
Particularly suited for text data, especially for tasks like spam detection or sentiment analysis.
Efficient and requires minimal training time.

- Process:
The TF-IDF features were passed into the Multinomial Naive Bayes model for training.
The model calculated the probability of a given text belonging to each category and selected the one with the highest probability.

- Evaluation:
Similar metrics like precision, recall, F1-score, and accuracy were used to evaluate performance.
Confusion matrix showed how well the model predicted each category.

![Classification Report for Naive Bayes](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/CR%20for%20Naive%20Bayes.png)

![Confusion Matrix for Naive Bayes](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/CM%20for%20Naive%20Bayes.png)

- Naive Bayes also has 98% accuracy. From confusion matrix, it is evident that entertainment category does very well whereas business category is little confused with technology but every category has very good accuracy.


### Recurrent Neural Network(RNN)

- Description: An RNN is a type of deep learning model designed to handle sequential data, such as text. It captures the relationships between words in a sequence, making it ideal for NLP tasks.

- Enhancements:
Used an embedding layer to convert words into dense vector representations.
Implemented a Bidirectional RNN to process the text in both forward and backward directions, enhancing context understanding.

- Process:The text was tokenized and padded to a fixed sequence length.
The RNN model consisted of an embedding layer, recurrent layers, and dense layers for classification.
Text Vectorization is implemented using Tensorflows.

- Evaluation:
The classification report showed the model's performance in terms of precision, recall, and F1-score.
The confusion matrix provided insights into the model's accuracy across all categories.

![Classification Report for RNN](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/CR%20for%20RNN.png)

![Confusion Matrix for RNN](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/CM%20for%20RNN.png)

- Recurrent Neural Network has 80% accuracy which is good but comparatively this is low. From confusion matrix, we can see that technology got confused with sports and entertainment. Also, entertainment got confused witg sportd and technology.Other three categories have good accuracy.

## Real and Synthesized examples

Some real examples from different categories is taken from Indian Express. Also, some synthesized examples were used to test each model.

#### Real Examples

- Hindi cinema’s best to worst in 2024: Of Pushpa 2 and Stree 2, Chamkila and Laapataa Ladies --->> Entertainment

- Nestlé judgement fallout: Switzerland suspends MFN clause in tax avoidance pact with India, could impact $100 bn investment commitment under EFTA deal --->> Business

- India vs Australia LIVE Cricket Score, 3rd Test: Rain stops play in Brisbane, AUS 19/0 vs IND --->> Sports

- US lawmakers tell Apple, Google to be ready to remove TikTok from app stores on January 19 --->> Technology

- Advancements in AI are transforming how technology is integrated into daily life --->> Technology

- Trump calls daylight saving time ‘very costly to Nation,’ demands it end --- Political

- How Bangladesh’s Constitution was adopted and what it says about minority rights --->> Research

- UPSC Essentials | Daily subject-wise quiz : International Relations MCQs on ‘Our Living Islands’ policy, Levant and more --->> Education

#### Synthesized Examples

- The company's profits increased by 30% this quarter --->> Business

- Online learning platforms are transforming education --->> Education

- An award-winning actor delivered a memorable performance --->> Entertainment

- The cricket match was a nail-biter till the last over --->> Sports

- Tech giants are unveiling new products at the annual conference --->> Technology



### Predictions Using Logistic Regression 

Real-World Examples:

- Text: Hindi cinema’s best to worst in 2024: Of Pushpa 2 and Stree 2, Chamkila and Laapataa Ladies
Predicted Category: entertainment

- Text: Nestlé judgement fallout: Switzerland suspends MFN clause in tax avoidance pact with India, could impact $100 bn investment commitment under EFTA deal
Predicted Category: business

- Text: India vs Australia LIVE Cricket Score, 3rd Test: Rain stops play in Brisbane, AUS 19/0 vs IND
Predicted Category: sports
 
- Text: US lawmakers tell Apple, Google to be ready to remove TikTok from app stores on January 19
Predicted Category: technology
 
- Text: Advancements in AI are transforming how technology is integrated into daily life.
Predicted Category: technology

- Text: Trump calls daylight saving time ‘very costly to Nation,’ demands it end.
Predicted Category: sports
 
- Text: How Bangladesh’s Constitution was adopted and what it says about minority rights.
Predicted Category: sports
 
- Text: UPSC Essentials | Daily subject-wise quiz : International Relations MCQs on ‘Our Living Islands’ policy, Levant and more.
Predicted Category: education

Synthesized Examples:

- Text: The company's profits increased by 30% this quarter.
Predicted Category: business

- Text: Online learning platforms are transforming education.
Predicted Category: education

- Text: An award-winning actor delivered a memorable performance.
Predicted Category: entertainment

- Text: The cricket match was a nail-biter till the last over.
Predicted Category: sports

- Text: Tech giants are unveiling new products at the annual conference.
Predicted Category: technology


Here, the model has predicted all the trained categories correctly in real world examples and also synthesized examples.Also, I have tested the model using two categories (Political, Research) just to check what the model is predicting, in logistic regression both those untrained categories are predicted as sports.

### Predictions Using Naive Bayes

Real-World Examples:

- Text: Hindi cinema’s best to worst in 2024: Of Pushpa 2 and Stree 2, Chamkila and Laapataa Ladies
Predicted Category: entertainment

- Text: Nestlé judgement fallout: Switzerland suspends MFN clause in tax avoidance pact with India, could impact $100 bn investment commitment under EFTA deal
Predicted Category: business

- Text: India vs Australia LIVE Cricket Score, 3rd Test: Rain stops play in Brisbane, AUS 19/0 vs IND
Predicted Category: sports

- Text: US lawmakers tell Apple, Google to be ready to remove TikTok from app stores on January 19
Predicted Category: technology

- Text: Advancements in AI are transforming how technology is integrated into daily life.
Predicted Category: technology

- Text: Trump calls daylight saving time ‘very costly to Nation,’ demands it end.
Predicted Category: business

- Text: How Bangladesh’s Constitution was adopted and what it says about minority rights.
Predicted Category: sports

- Text: UPSC Essentials | Daily subject-wise quiz : International Relations MCQs on ‘Our Living Islands’ policy, Levant and more.
Predicted Category: education


Synthesized Examples:

- Text: The company's profits increased by 30% this quarter.
Predicted Category: business

- Text: Online learning platforms are transforming education.
Predicted Category: education

- Text: An award-winning actor delivered a memorable performance.
Predicted Category: entertainment

- Text: The cricket match was a nail-biter till the last over.
Predicted Category: sports

- Text: Tech giants are unveiling new products at the annual conference.
Predicted Category: technology

Naive Bayes also predicted all the trained categories correctly for both examples. In this model, the untrained categories political and reasearch has been identified as business and sports.


### Predictions Using RNN 

Real-World Examples:

- Text: Hindi cinema’s best to worst in 2024: Of Pushpa 2 and Stree 2, Chamkila and Laapataa Ladies
Predicted Category: entertainment

- Text: Nestlé judgement fallout: Switzerland suspends MFN clause in tax avoidance pact with India, could impact $100 bn investment commitment under EFTA deal
Predicted Category: business

- Text: India vs Australia LIVE Cricket Score, 3rd Test: Rain stops play in Brisbane, AUS 19/0 vs IND
Predicted Category: sports

- Text: US lawmakers tell Apple, Google to be ready to remove TikTok from app stores on January 19
Predicted Category: technology

- Text: Advancements in AI are transforming how technology is integrated into daily life.
Predicted Category: technology

- Text: Trump calls daylight saving time ‘very costly to Nation,’ demands it end.
Predicted Category: sports

- Text: How Bangladesh’s Constitution was adopted and what it says about minority rights.
Predicted Category: sports

- Text: UPSC Essentials | Daily subject-wise quiz : International Relations MCQs on ‘Our Living Islands’ policy, Levant and more.
Predicted Category: education


Synthesized Examples:

- Text: The company's profits increased by 30% this quarter.
Predicted Category: business

- Text: Online learning platforms are transforming education.
Predicted Category: education

- Text: An award-winning actor delivered a memorable performance.
Predicted Category: entertainment

- Text: The cricket match was a nail-biter till the last over.
Predicted Category: sports

- Text: Tech giants are unveiling new products at the annual conference.
Predicted Category: technology

RNN has also done good with the trained categories even though the accuracy was comparatively low, so it has also identified the untrained categories as sports like Logistic Regression.

### Zero-Shot Classification

- Description: Zero-shot classification eliminates the need for labeled training data. Using the facebook/bart-large-mnli model from Hugging Face, this approach directly predicts the category of a given text based on predefined candidate labels.

- Advantages:
Ideal for scenarios where labeled data is unavailable.
Provides a flexible approach to classify text into any set of categories.

- Process:
The model takes the text and candidate labels as input.
It computes the likelihood of the text belonging to each category and selects the most probable one.

- Evaluation:
Demonstrated with real-world examples and synthesized examples.
Predictions were compared with true labels for evaluation.

![Classification Report for Zero-Shot Classification](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/CR%20for%20Zero-shot.png)

![Confusion Matrix for Zero-Shot Classification](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_HemamaliniVV_ShifatNowrin/blob/main/CM%20for%20Zero-Shot.png)

### Prediction using Zero-Shot Classification

Real-world Examples: 

- Text: Hindi cinema’s best to worst in 2024: Of Pushpa 2 and Stree 2, Chamkila and Laapataa Ladies
Predicted Category: entertainment

- Text: Nestlé judgement fallout: Switzerland suspends MFN clause in tax avoidance pact with India, could impact $100 bn investment commitment under EFTA deal
Predicted Category: business

- Text: India vs Australia LIVE Cricket Score, 3rd Test: Rain stops play in Brisbane, AUS 19/0 vs IND
Predicted Category: sports

- Text: US lawmakers tell Apple, Google to be ready to remove TikTok from app stores on January 19
Predicted Category: technology

- Text: UPSC Essentials | Daily subject-wise quiz : International Relations MCQs on ‘Our Living Islands’ policy, Levant and more.
Predicted Category: education

Synthesized Examples:

- Text: The company's profits increased by 30% this quarter.
Predicted Category: business

- Text: Online learning platforms are transforming education.
Predicted Category: education

- Text: An award-winning actor delivered a memorable performance.
Predicted Category: entertainment

- Text: The cricket match was a nail-biter till the last over.
Predicted Category: sports

- Text: Tech giants are unveiling new products at the annual conference.
Predicted Category: technology

Zero-shot classification has 50% accuracy but it seems like it has done good job in predicting the categories of both real-world and synthesized examples, but has some confusion between the categories.


### Evaluation Metrics
 
Each model was evaluated using the following metrics:

- Accuracy: Overall correctness of predictions.

- Precision: Ability to avoid false positives.

- Recall: Ability to identify all true positives.

- F1-Score: Harmonic mean of precision and recall.

- Confusion Matrix: Visualization of true vs. predicted labels

| Model                    | Accuracy | F1-Score |
|--------------------------|----------|----------|
| Logistic Regression      | 98%      | 0.99     |
| Naive Bayes              | 98%      | 0.98     |
| RNN                      | 80%      | 0.80     |
| Zero-Shot Classification | 50%      | 0.50     |

Logistic Regression and Naive Bayes are the best model here with high accuracy , however all the models has predicted the examples give correctly according to their respective categories.


#### Technologies Used

Python: Primary programming language.

Jupyter Notebook: For code execution and analysis.

Libraries:

- TensorFlow: For RNN implementation.

- Hugging Face Transformers: For Zero-Shot Classification.

- Scikit-learn: For Logistic Regression, Naive Bayes, and evaluation metrics.

- Pandas, NumPy: For data manipulation and numerical operations.
Matplotlib, Seaborn: For data visualization.

## Conclusion 

This project demonstrates the application of various NLP models for news classification. While traditional models like Logistic Regression and Naive Bayes provide competitive performance, deep learning models like RNNs excel in capturing context from text. Zero-shot classification offers a versatile option when labeled data is unavailable.

Through this project, we observed that the choice of model significantly impacts the accuracy and generalizability of the classification task. Preprocessing techniques such as tokenization and feature extraction (e.g., TF-IDF) played a critical role in improving model performance. Deep learning models required more computational resources but delivered superior results by leveraging word embeddings and sequential context. Finally, the inclusion of Zero-shot classification demonstrated the potential of pre-trained transformer models for real-world scenarios where data labeling is impractical or expensive.



































