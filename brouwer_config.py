modelID = '5a'

context_size = 200
retrieval_size = 80

trainingsuf = 'br-origfulldutch'
# trainingsuf = 'br-holdoutB-dutch'
#trainingsuf = 'br-nonsterdutch8k'

traincode='full'

maxup = 7000 #7000 in Brouwer
itperup = 100 #100 in Brouwer

reducelr = False

embdic='embs/brouwerCOALS-100.txt'
binary = True

tryloc=True

# dict='brouwerGloVe-100.txt'
# trainingsuf = 'br-origfulleng'

notes='original replication settings but trying with word2loc as input to integration layers'
