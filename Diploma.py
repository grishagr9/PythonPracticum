#Удаление знаков пунктуации
import re
import os

write = open("push_step1.txt","w")
with open("krug.txt", "r") as file1:
    for line in file1:
      write.write(re.sub(r'[^\w\s]','', line))
write.close()

#Применение лемматизатора
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lmtzr = WordNetLemmatizer()

write = open("push_step2.txt","w")
with open("push_step1.txt", "r") as file1:
    for line in file1:
        for word in line.strip().split():
          write.write(lmtzr.lemmatize(word)+' ')
        write.write("\n")
write.close()

print(lmtzr.lemmatize('dogs'))

#Удаление стоп-слов
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
def delete_stop_words(words):
    return [word for word in words.split() if word.lower() not
            in stopwords.words('russian')]

file = open("push_step2.txt","r").read()
text1 = delete_stop_words(file)
write = open("push_step3.txt","w")
for word in text1:
    write.write(word+" ")
write.close()
# Importing necessary libraries
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatiser = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('stopwords')
# Defining a module for Text Processing
def text_process(tex):
    # 1. Removal of Punctuation Marks
    nopunct=[char for char in tex if char not in string.punctuation]
    nopunct=''.join(nopunct)
    # 2. Lemmatisation
    a=''
    i=0
    for i in range(len(nopunct.split())):
        b=lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a=a+b+' '
    # 3. Removal of Stopwords
    return [word for word in a.split() if word.lower() not
            in stopwords.words('english')]
x=[]
y=[]
file = open("Esenin.txt").read().split("0")

for line in file:
    x.append(line)
    y.append(1)

file = open("Pushkin.txt").read().split("1")

for line in file:
    x.append(line)
    y.append(0)

#print(x)
#print(y)
#print(len(x))
#print(len(y))

# Importing necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# 80-20 splitting the dataset (80%->Training and 20%->Validation)
X_train, X_test, y_train, y_test = train_test_split(x, y
                                  ,test_size=0.2, random_state=1234)

# defining the bag-of-words transformer on the text-processed corpus # i.e., text_process() declared in II is executed...
bow_transformer=CountVectorizer(analyzer=text_process).fit(X_train)
# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_train=bow_transformer.transform(X_train)#ONLY TRAINING DATA
# transforming into Bag-of-Words and hence textual data to numeric..
text_bow_test=bow_transformer.transform(X_test)#TEST DATA

#print(X_train)
#print(X_test)

# Importing necessary libraries
from sklearn.naive_bayes import MultinomialNB
# instantiating the model with Multinomial Naive Bayes..
model = MultinomialNB()
# training the model...
model = model.fit(text_bow_train, y_train)
print("y train")
print(model.score(text_bow_train, y_train))
print("y test")
print(model.score(text_bow_test, y_test))

# Importing necessary libraries
from sklearn.metrics import classification_report

# getting the predictions of the Validation Set...
predictions = model.predict(text_bow_test)
# getting the Precision, Recall, F1-Score
print(classification_report(y_test,predictions))


# Importing necessary libraries
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
# Defining a module for Confusion Matrix...
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0])
                                  , range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm = confusion_matrix(y_test,predictions)
plt.figure()
plot_confusion_matrix(cm, classes=[0,1], normalize=True,
                      title='Confusion Matrix')

from sklearn.feature_extraction.text import CountVectorizer
text = []
# получим объект файла
file1 = open("a.txt", "r")

while True:
    # считываем строку
    line = file1.readline()
    # прерываем цикл, если строка пустая
    if not line:
        break
    # выводим строку
    text.append(line.strip())

# закрываем файл
file1.close

vectorizer = CountVectorizer()
bag_wrd = vectorizer.fit_transform(text)
#Получаем матрицу:
bag_wrd.toarray()
features = vectorizer.get_feature_names_out()
print(features)
print(bag_wrd.toarray())

#Посчитать распределение различных знаков препинания в тексте
def punctuation_freq(lower_text):
    symbols = [',', '.', '?', '!', ':', ';', '(', '–', '"', "'"]
    punct_dict = dict.fromkeys(symbols, 0) # ',.?!:;(–"\''
    if len(lower_text):
       for symbol in symbols:
         punct_dict[symbol] = lower_text.count(symbol)
       punct_dict['"'] /= 2
       punct_dict["'"] /= 2
    else:
       return list(punct_dict.values())
    return list(punct_dict.values())
punctuation_freq('fewepofkwkm;-()ef')

#распределение частей речи в тексте
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
POS = ['NOUN', 'ADJF', 'ADJS','COMP','VERB','INFN','PRTF','PRTS','GRND', 'ADVB ','NPRO','PRED','PREP','CONJ','PRCL','INTJ']
def POS_distribution(tokenized):
    words_POS = [morph.parse(word)[0].tag.POS for word in tokenized]
    POS_distr = dict.fromkeys(POS, 0) # инициализация dict для всех частей реч
    POS_counted = dict(collections.Counter(words_POS))
    for word in POS:
       if word in POS_counted.keys():
          POS_distr[word] = POS_counted[word]
    return list(POS_distr.values())