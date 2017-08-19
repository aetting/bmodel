import pickle
from util_brouwer import *

# meaning_dict_list = read_meaning_dict('sim1-d.csv',dutch=True)
# inputpairs,origid2meaning = generate_brouwer_train_sentences(meaning_dict_list,dutch=True)
# print len(inputpairs)
# # 
# trainingsuf = 'br-origfulldutch'
# 
# with open('trainingpairs-%s'%trainingsuf,'w') as trainingfile: pickle.dump(inputpairs,trainingfile,pickle.HIGHEST_PROTOCOL)
# with open('trainingpairs-%s'%trainingsuf) as trainingfile: trainingpairs = pickle.load(trainingfile)
# 

trainingsufnew = 'br-stereng8k'
# trainingpairsnew = trainingpairs[8000:]

# with open('trainingpairs-%s'%trainingsufnew,'w') as trainingfilenew: pickle.dump(trainingpairsnew,trainingfilenew,pickle.HIGHEST_PROTOCOL)
with open('trainingpairs-%s'%trainingsufnew) as trainingfile: trainingpairs = pickle.load(trainingfile)
for p in trainingpairs: print p
print len(trainingpairs)
# 
# dict = 'coals-svdb-100.model'
# # dict='/Users/allysonettinger/Desktop/meaning_cc/modeling/models/pretrained_embeddings/glove/glove-Wik-Gig/glove.6B.100d.txt'
# 
# word2id = make_word2id(trainingpairs)
# # print word2id
# load_word2dist(dict,word2id,binary=False,debug=False,filename='brouwer2COALS-100.txt',delim=',')

# simsents = generate_hoeks(meaning_dict_list,dutch=True)
# for s in simsents: print s

