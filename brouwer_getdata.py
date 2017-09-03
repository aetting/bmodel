import pickle
from util_brouwer import *

ster_scale=.5
meaning_dict_list,triplets = read_meaning_dict('sentsources/sim1-d.csv',dutch=True)
inputpairs,origid2meaning = generate_brouwer_train_sentences(meaning_dict_list,triplets,dutch=True,holdout=False,ster_scale=ster_scale)
# with open('textcheck.txt','w') as out:
#     for s,_,_ in inputpairs: out.write(' '.join(s) + '\n')
# # 
trainingsuf = 'br-orig-ster%s-dutch'%ster_scale
# 
with open('trainingpairs/trainingpairs-%s'%trainingsuf,'w') as trainingfile: pickle.dump(inputpairs,trainingfile,pickle.HIGHEST_PROTOCOL)
with open('trainingpairs/trainingpairs-%s'%trainingsuf) as trainingfile: trainingpairs = pickle.load(trainingfile)
with open('textcheck.txt','w') as out:
    for s,_,_ in trainingpairs: out.write(' '.join(s) + '\n')
print len(trainingpairs)

# trainingsufnew = 'br-stereng8k'
# trainingpairsnew = trainingpairs[8000:]

# with open('trainingpairs/trainingpairs-%s'%trainingsufnew,'w') as trainingfilenew: pickle.dump(trainingpairsnew,trainingfilenew,pickle.HIGHEST_PROTOCOL)
# with open('trainingpairs/trainingpairs-%s'%trainingsufnew) as trainingfile: trainingpairs = pickle.load(trainingfile)
# for p in trainingpairs: print p
# print len(trainingpairs)
# 
# dict = 'coals-svdb-100.model'
# # dict='/Users/allysonettinger/Desktop/meaning_cc/modeling/models/pretrained_embeddings/glove/glove-Wik-Gig/glove.6B.100d.txt'
# 
# word2id = make_word2id(trainingpairs)
# # print word2id
# load_word2dist(dict,word2id,binary=False,debug=False,filename='brouwer2COALS-100.txt',delim=',')

# simsents = generate_hoeks(meaning_dict_list,dutch=True)
# for s in simsents: print s

