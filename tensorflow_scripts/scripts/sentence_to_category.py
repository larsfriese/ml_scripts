from numpy import array
import tensorflow, datetime, pandas, twint
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, Dropout
from collections import Counter

# max length is the length of a single string, vocab size the maximum amount of words learned
max_length = 60
vocab_size = 400
lookup_input = {}

def train(data_csv, string_column, category_columns):

    df = pandas.read_csv(data_csv)
    docs_x = df[string_column].tolist()
    docs_y= []
    for index, row in df.iterrows():
        temp = []
        for i in category_columns:
            temp.append(row[i])
        docs_y.append(temp)

    # define documents
    # y = array([[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0]])

    # sad, andgry, scared/feared, happy, surprised, disgusted
    # [0, 0, 0, 0, 0, 0]

    def encode_matrix(list_of_strings):
        encoded_docs = [one_hot(d, vocab_size) for d in list_of_strings]
        for i,x in zip(encoded_docs, docs_x):
            lookup_input[x]=i
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        return padded_docs

    x = encode_matrix(docs_x)
    y = array(docs_y)

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))

    # define optimizer, loss, epochs and data
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=100, verbose=0)

    # Evaluation:
    loss, accuracy = model.evaluate(x, y, verbose=0)
    print('Accuracy: %f /100' % (accuracy*100))

    # save model
    model.save('models/ml_model_stc', '/')

def predict(strings):

    def encode_matrix(list_of_strings):
        encoded_docs = [one_hot(d, vocab_size) for d in list_of_strings]
        for i,x in zip(encoded_docs, strings):
            lookup_input[x]=i
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        return padded_docs

    # load model
    model = tensorflow.keras.models.load_model('models/ml_model_stc')

    # predict the strings
    results = model.predict(encode_matrix(strings)).tolist()

    # count up results
    opinions=[]
    for i in results:
        n=i.index(max(i))
        if n==0:
            opinions.append('sad')
        elif n==1:
            opinions.append('angry')
        elif n==2:
            opinions.append('scared')
        elif n==3:
            opinions.append('happy')
        elif n==4:
            opinions.append('surprised')
        elif n==5:
            opinions.append('disgusted')

    print(Counter(opinions))

if __name__ == '__main__':
    train('../data/sentence_to_category_data.csv', 'tweet', ['sad', 'angry', 'scared', 'happy', 'surprised', 'disgusted'])
    print(predict(['this is a test #1','tweet #2','tweet #3','tweet #4']))
