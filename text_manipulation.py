import spacy
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

nlp = spacy.load('en', disable=['parser', 'ner'])


def clean_text(text, use_pronoun_token=False, lower=True):
    """Covert text to cleaned format to make it more interpretable for NLP models.

    # Arguments
        text: Input string
        use_pronoun_token: whether to use 'PRONOUN' token in place of pronouns or keep them as it is
        lower: boolean to whether to make text lowercase

    # Returns
        cleaned text
    """
    if lower:
        tokens = nlp(text.lower())
    else:
        tokens = nlp(text)
    words = []
    for token in tokens:
        lemma = token.lemma_

        # in spacy pronouns(you/me/he/she/his/him/they/them...etc) are lemmatised as '-PRON-'
        if lemma == '-PRON-':

            # if we want to use token PRONOUN in place of pronouns
            if use_pronoun_token:
                words.append("PRONOUN")

            # using original pronouns as it is
            else:
                words.append(str(token))

                # ignoring numbers & lemmas having presence of any other than alphanumeric character
        elif not (re.search("[^a-z0-9]", lemma) or lemma.isnumeric()):
            words.append(lemma)

    corpus = " ".join(words)
    return corpus


class TextProcessing:
    """Class provide easy tool to implement different steps in preprocessing of text column and label column.
      DataFrame is created for current instance from given dataframe and columns are built while processing"""

    def __init__(self, data, text_col=None, label_col=None, is_training=True):
        """Initialising data to be processed and other needful things to check/initialised at the start"""

        self.is_training = is_training
        self.label_col = label_col
        self.text_col = text_col

        # data checks, df creation to use & parameters setting for train set
        if self.is_training:
            print("Data loaded !", )

            self.data = data[[self.text_col, self.label_col]]

            self.classes = self.data[label_col].unique()
            self.classes.sort()
            print(f"{len(self.classes)} classes in labels: {', '.join(self.classes)}")

            if self.check_null():
                self.visualise_class_distribution()

        # df creation for test set
        else:
            self.data = data[[self.text_col]]

        # total samples in given df
        self.total_samples = len(self.data)

        # original text length(i.e no of words) will be useful
        self.data['text_length'] = self.data['text'].apply(lambda x: len(x.split()))

        # creating corpus column
        self.data['corpus'] = None

    def create_corpus(self, use_pronoun_token=False, lower=True):
        """Creates corpus by cleaning each text"""
        self.data['corpus'] = self.data[self.text_col].apply(lambda x: clean_text(x, use_pronoun_token, lower))
        self.data['corpus_length'] = self.data['corpus'].apply(lambda x:len(x.split()))
        if self.is_training:
            print("Corpus created successfully")

    def check_null(self):
        """Checks for null as soon as data is loaded"""

        print("\nChecking for null values...")

        # column wise nulls
        col_nulls = self.data[[self.text_col, self.label_col]].isna().sum()

        if not all(col_nulls):
            print("No null values in any feature and label columns of dataset")
            return True
        else:
            for index in col_nulls.index:
                if col_nulls[index] != 0:
                    print(index)
            print("Above column/columns contain null values")

    def visualise_class_distribution(self, test_size=None, normalize=True):
        """Visualise class wise distribution in labels of whole/test-test dataset"""

        if test_size:
            self.y_train, self.y_test = train_test_split(self.data[self.label_col], test_size=test_size,
                                                         random_state=100)

            plt.figure(figsize=(10, 5))

            _, counts = np.unique(self.y_train.values, return_counts=True)
            lines = []
            for i in range(len(counts)):
                lines.append(self.classes[i] + ' ' + str(counts[i]))
            text = "\n".join(lines)

            plt.subplot(1, 2, 1)
            self.y_train.value_counts(normalize=normalize).plot(kind='bar')
            plt.title('Train set')
            plt.legend([text])

            _, counts = np.unique(self.y_test.values, return_counts=True)
            lines = []
            for i in range(len(counts)):
                lines.append(self.classes[i] + ' ' + str(counts[i]))
            text = "\n".join(lines)

            plt.subplot(1, 2, 2)
            self.y_test.value_counts(normalize=normalize).plot(kind='bar')
            plt.title('Test set')
            plt.legend([text])
            plt.show()

        else:
            _, counts = np.unique(self.data[self.label_col].values, return_counts=True)

            lines = []
            for i in range(len(counts)):
                lines.append(self.classes[i] + ' ' + str(counts[i]))
            text = "\n".join(lines)

            self.data[self.label_col].value_counts(normalize=normalize).plot(kind='bar')
            plt.title('Whole dataset')
            plt.legend([text])
            plt.show()

    def encode_labels(self, using_sklearn_model=True, ohe_sparse=False):
        """Encodes labels in binary / one-hot-code fashion taking into account"""

        le = LabelEncoder()
        ohe = OneHotEncoder(sparse=ohe_sparse)

        # if we are using sklearn classificationmodels for data with more than 2 then it doesn't
        # need labels in one-code-fashion
        if len(self.classes) == 2 or using_sklearn_model:
            self.y = le.fit_transform(self.data[self.label_col])
            return le
        else:
            self.y = ohe.fit_transform(self.data[self.label_col].values.reshape(-1, 1))
            return ohe

    def view_corpus(self, text_length_order=None, no_of_samples=10):
        """Displays corpus created for given texts in samples"""

        # if corpus is created
        if self.data['corpus'] is not None:
            columns = [self.label_col, self.text_col, 'corpus']

            # sort according to text length
            if text_length_order == 'ASC':
                view = self.data.sort_values(by='text_length')[columns].values

            elif text_length_order == 'DESC':
                view = self.data.sort_values(by='text_length', ascending=False)[columns].values

            else:
                view = self.data[columns].values

            for i in range(no_of_samples):
                print()
                print('Label:', view[i][0])
                print('Original text:', view[i][1])
                print('Reduced text:', view[i][2])
        else:
            print("Corpus yet not created")

    def create_final_features(self, max_words=None, final_features='bow_matrix', max_seq_len = 64, trained_tokenizer=None,
                              mode='tfidf', padding='post'):
        """Creates word matrix using corpus"""

        if self.is_training:
            # creating custom tokenizer from keras
            tokenizer = Tokenizer(num_words=max_words, lower=False)

            # fitting tokenizer on texts
            tokenizer.fit_on_texts(self.data['corpus'])

            # if we want to use BoW matix as final features i.e X
            if final_features == 'bow_matrix':
                self.X = tokenizer.texts_to_matrix(self.data['corpus'], mode=mode)
                print(f"Matrix using corpora is built with '{mode}' mode")
                print("Total words in corpora:", len(tokenizer.word_index))
                print("Shape of bow matrix:", self.X.shape)

                # retuninzer tokenizer and mode so can be used later on while prediction on test set
                return tokenizer, mode

            # if final features are padded sequences with given length
            elif final_features == 'padded_sequences':
                sequences = tokenizer.texts_to_sequences(self.data['corpus'])
                padded_seq = pad_sequences(sequences=sequences, maxlen=max_seq_len, padding=padding)
                self.X = padded_seq
                print(f"Sequences with length {max_seq_len} created")

                # returning tokizer so can be used at the deployment of model or prediction on test set
                return tokenizer
        else:
            if final_features == 'bow_matrix':
                self.X = trained_tokenizer.texts_to_matrix(self.data['corpus'], mode=mode)
            elif final_features == 'padded_sequences':
                sequences = trained_tokenizer.texts_to_sequences(self.data['corpus'])
                padded_seq = pad_sequences(sequences=sequences, maxlen=max_seq_len, padding=padding)
                self.X = padded_seq

    def split_data(self, test_size=0.2):
        """Returns train & test splits for created X and y"""

        # visualise splits created
        self.visualise_class_distribution(test_size=test_size)

        return train_test_split(self.X, self.y, test_size=test_size, random_state=100)
