import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import collections
from sklearn.feature_extraction import DictVectorizer
import keras
import keras.layers as L
from sklearn.model_selection import train_test_split
from keras.models import model_from_json



def main():
    min_count = 10
    test_size =0.2
    model_path = "./model.json"
    model_weight = "./model.h5"
    train,test = getData()
    transformed_train,vectorizer = preprocess(train)
    transformed_test,vect_test = preprocess_test(test)
    token_id = preprocess_token(transformed_train,min_count)
    #data_train,data_val = split_data(train,test_size)
    #model = model_train(token_id,vectorizer,data_train,data_val)

    model = load(model_path,model_weight)
    for i in range(len(transformed_test)):
        testd = transformed_test.iloc[i,:]
        batch = make_batch(token_id,vectorizer,testd,max_len=None,word_dropout=0)
        print(model.predict(batch))



def load(model_path,model_weight):
    json_file = open(model_path,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weight)
    return loaded_model

def getData():
    #!curl -L https://www.dropbox.com/s/5msc5ix7ndyba10/Train_rev1.csv.tar.gz?dl=1 -o Train_rev1.csv.tar.gz
    #!tar -xvzf ./Train_rev1.csv.tar.gz
    train_data = pd.read_csv("./Train_rev1.csv", index_col=None)[1:10000]
    test_data = pd.read_csv("./Test_rev1.csv", index_col=None)
    return train_data,test_data

def gettokenstring(data):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    preprocess = lambda text: ' '.join(tokenizer.tokenize(str(text).lower()))
    data['FullDescription'] = data['FullDescription'].apply(lambda x : preprocess(x))
    data['Title'] = data['Title'].apply(lambda x : preprocess(x))
    return data

def preprocess(data):
    data['Log1pSalary'] = np.log1p(data['SalaryNormalized']).astype('float32')
    text_columns = ["Title", "FullDescription"]
    categorical_columns = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]
    #fill the missing values
    data[categorical_columns] = data[categorical_columns].fillna('NaN') # cast missing values to string "NaN"
    top_companies,top_counts = zip(*collections.Counter(data['Company']).most_common(1000))
    recognized_companies = set(top_companies)
    #only use top 1000 companies
    data["Company"] = data["Company"].apply(lambda comp: comp if comp in recognized_companies else "Other")
    categorical_vectorizer = DictVectorizer(dtype=np.float32,sparse=False)
    categorical_vectorizer.fit(data[categorical_columns].apply(dict,axis=1))
    data = gettokenstring(data)
    return data,categorical_vectorizer
def preprocess_test(data):
    text_columns = ["Title", "FullDescription"]
    categorical_columns = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]
    #fill the missing values
    #data[categorical_columns] = data[categorical_columns].fillna('NaN') # cast missing values to string "NaN"
    #top_companies,top_counts = zip(*collections.Counter(data['Company']).most_common(1000))
#    recognized_companies = set(top_companies)
    #only use top 1000 companies
    categorical_vectorizer = DictVectorizer(dtype=np.float32,sparse=False)
    categorical_vectorizer.fit(data[categorical_columns].apply(dict,axis=1))
    data = gettokenstring(data)
    return data,categorical_vectorizer
# Count how many times does each token occur in both "Title" and "FullDescription" in total
# build a dictionary { token -> it's count }
def countnum(data):
    token_counts = collections.Counter()
    for i in range(len(data)):
        if (i % 5000 == 0):
            print(i)
        s_title = str(data['Title'].values[i]).split(" ")
        s_description = str(data['FullDescription'].values[i]).split(" ")
        for word in s_title:
            token_counts[word] += 1
        for word in s_description:
            token_counts[word] += 1
    return token_counts

#get the tokens that satisfy the need
def filter(token_counts,min_count):
    tokens = [k for k,v in token_counts.items() if v >= min_count]
    return tokens

def preprocess_token(data,min_count=10):
# Add a special tokens for unknown and empty words
    token_counts = countnum(data)
    tokens = filter(token_counts,min_count)
    UNK, PAD = "UNK", "PAD"
    tokens = [UNK, PAD] + sorted(tokens)
    token_to_id = {tokens[i]:i for i in range(len(tokens))}
    return token_to_id

def as_matrix(token_to_id,sequences,max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    UNK_IX,PAD_IX = map(token_to_id.get,['UNK','PAD'])
    if isinstance(sequences[0],str):
        sequences = list(map(str.split,sequences))
    max_len = min(max(map(len,sequences)),max_len or float('inf'))

    matrix = np.full((len(sequences),max_len),np.int32(PAD_IX))
    for i,seq in enumerate(sequences):
        row_ix = [token_to_id.get(word,UNK_IX) for word in seq[:max_len]]
        matrix[i,:len(row_ix)] = row_ix
    return matrix




#split the datasets to train_test datasets
def split_data(data,test_size):
    data_train, data_val = train_test_split(data, test_size=0.2, random_state=42)
    data_train.index = range(len(data_train))
    data_val.index = range(len(data_val))
    return data_train,data_val
#most times, we cannot just split batches in training, so we could do batch preprocess manually
def make_batch(token_to_id,categorical_vectorizer,data, max_len=None, word_dropout=0):
    #form a batch dictionary
    UNK_IX,PAD_IX = map(token_to_id.get,['UNK','PAD'])
    batch = {}
    batch['Title'] = as_matrix(token_to_id,data['Title'].values,max_len)
    batch['FullDescription'] = as_matrix(token_to_id,data['FullDescription'].values,max_len)
    #split the columns into different types
    text_columns = ["Title", "FullDescription"]
    categorical_columns = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]
    target_column = "Log1pSalary"

    batch['Categorical'] = categorical_vectorizer.transform(data[categorical_columns].apply(dict, axis=1))

    if word_dropout!=0:
        batch["FullDescription"] = apply_word_dropout(batch["FullDescription"],1.-word_dropout,UNK_IX,PAD_IX)
    if target_column in data.columns:
        batch[target_column] = data[target_column].values
    return batch

def apply_word_dropout(matrix,keep_prob,replace_with, pad_ix):
    dropout_mask = np.random.choice(2,np.shape(matrix),p=[keep_prob,1-keep_prob])
    dropout_mask &= matrix != pad_ix
    return np.choose(dropout_mask,[matrix,np.full_like(matrix,replace_with)])

#3 branches for title/description/categories respectively
#build model structure
def build_model(tokens,categorical_vectorizer,hid_size=64):
    """ Build a model that maps three data sources to a single linear output: predicted log1p(salary) """
    n_cat_features = len(categorical_vectorizer.vocabulary_)
    n_tokens = len(tokens)
    l_title = L.Input(shape=[None],name="Title")
    l_descr = L.Input(shape=[None],name="FullDescription")
    l_categ = L.Input(shape=[n_cat_features],name="Categorical")
    # Build your monster!

    # <YOUR CODE>
    # Embedding (size_vocabulary,embeding_dim,max_length)
    embedding_dim = 64
    vocabulary_size = n_tokens
    num_filters = 32
    max_length = 10
    nb_filter = 16
    # embedding layer
    title_embedding = L.Embedding(input_dim=n_tokens,output_dim=embedding_dim)(l_title)
    descr_embedding = L.Embedding(input_dim=n_tokens,output_dim=embedding_dim)(l_descr)

    # convLayer
    title_conv = L.Conv1D(nb_filter,max_length,activation='relu')(title_embedding)
    descr_conv = L.Conv1D(nb_filter,max_length,activation='relu')(descr_embedding)

    # MaxPooling Layer
    title_pool = L.GlobalMaxPool1D()(title_conv)
    descr_pool = L.GlobalMaxPool1D()(descr_conv)

    # output_layer_title = L.Conv1D(1,kernel_size = 3)(l_title)
    output_layer_categ = L.Dense(hid_size)(l_categ)

    concat = L.Concatenate(axis=-1)([title_pool,descr_pool,output_layer_categ])
    # output_layer2 = L.Dense(hid_size)(output_layer_categ)
    output_layer3 = L.Dense(1)(concat)
    # end of your code

    model = keras.models.Model(inputs=[l_title,l_descr,l_categ],outputs=[output_layer3])
    model.compile('adam','mean_squared_error',metrics=['mean_absolute_error'])
    return model

#produce the batches of the datasets
def iterate_minibatches(token_to_id,vect,data,batch_size=256,shuffle=True,cycle=False,**kwargs):
    target_column = "Log1pSalary"

    while True:
        indices = range(len(data))
        if shuffle:
            indices = np.random.permutation(indices)

        for start in range(0,len(indices),batch_size):
            batch = make_batch(token_to_id,vect,data.iloc[indices[start:start+batch_size]],**kwargs)
            target = batch.pop(target_column)
            yield batch,target
        if not cycle:break

def model_train(tokens,categorical_vectorizer,data_train,data_val,batch_size=256,epochs=2,steps_per_epoch=100):
#model training
    model = build_model(tokens,categorical_vectorizer)
    model.fit_generator(iterate_minibatches(tokens,categorical_vectorizer,data_train,batch_size,cycle=True,word_dropout=0.05),
                    epochs=epochs,steps_per_epoch=steps_per_epoch,
                    validation_data=iterate_minibatches(tokens,categorical_vectorizer,data_val,batch_size,cycle=True),
                    validation_steps=data_val.shape[0] // batch_size
                    )
#training and evaluation

##yield::return a generator
    model_json = model.to_json()
    with open("model.json","w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("model.h5")
    print("save model to HDF5")
    return model

#print metrics
def print_metrics(model,data,batch_size,name="",**kw):
    squared_error = abs_error = num_samples = 0.0
    for batch_x,batch_y in iterate_minibatches(data,batch_size=batch_size,shuffle=False,**kw):
        batch_pred = model.predict(batch_x)[:,0]
        squared_error += np.sum(np.square(batch_pred - batch_y))
        abs_error += np.sum(np.abs(batch_pred - batch_y))
        num_samples += len(batch_y)
    print("%s results:" % (name or ""))
    print("Mean square error: %.5f" % (squared_error / num_samples))
    print("Mean absolute error: %.5f" % (abs_error / num_samples))
    return squared_error,abs_error


def predict_new(data_test,model):
    return model.predict(data_test)

if __name__ == '__main__':
    main()