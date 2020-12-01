from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from text_manipulation import *
import pandas as pd
import joblib

# using objects pickled while training of data
with open("DataProcessingGloVe.pkl", "rb") as file:
    trained_pronoun_token = joblib.load(file)
    trained_tokenizer = joblib.load(file)
    max_seq_len = joblib.load(file)
    class_names = joblib.load(file)

# loading embedding layer of neural network
with open("GloveEmbeddingVectors.pkl", "rb") as file:
    embedding_vectors = joblib.load(file)


def deep_nn_model(lstm_units=64, pretrained_embedding_layer=None, total_words=None, embed_dims=None, max_seq_len=None,
                  loss='categorical_crossentropy', optimizer='adam'):
    """Function returns deep neural network with first layer as embedding layer can be useful for sequential data"""

    model = Sequential()

    if pretrained_embedding_layer:
        model.add(pretrained_embedding_layer)
    else:
        model.add(Embedding(input_dim=total_words + 1, output_dim=embed_dims, input_length=max_seq_len))
    model.add(Bidirectional(LSTM(units=lstm_units, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def predict_sentiment(model, texts):
    """Inferencing on any text dataset"""

    # performing same set of operations on texts as on training data
    data = pd.DataFrame({'text':texts})
    test_preprocess = TextProcessing(data, text_col='text', is_training=False)
    test_preprocess.create_corpus(use_pronoun_token=trained_pronoun_token)
    test_preprocess.create_final_features(trained_tokenizer=trained_tokenizer, final_features='padded_sequences', max_seq_len=32)
    X = test_preprocess.X
    results = model.predict(X)
    results = np.where(results >= 0.5, 1, 0)
    results = results.flatten()
    results = [class_names[result] for result in results]
    return results


embedding_layer = Embedding(len(trained_tokenizer.word_index)+1, 100, weights=[embedding_vectors], input_length=max_seq_len, trainable=False)
glove_model = deep_nn_model(pretrained_embedding_layer=embedding_layer, loss='binary_crossentropy', lstm_units=128)
glove_model.load_weights('sentiment_glove.hdf5')