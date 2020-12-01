# Steps performed for Text classification

1. Cleaned text in the dataset and created corpus from it.
2. Choosing transormation of corpus to Bag of Words (BoW) matrix or padded sequence.
3. Traditional classification model such as Logistic Regression, Random Forest and XGBoost trained on BoW matrix
4. Word embedding methods are trained on padded sequences created out of texts such GloVe word embedding.
5. Using State of the art BERT for classifying texts, it is heavy model so inference time on trained BERT classification model takes is quite higher than other methods so trade off between time computation & accuracy 


