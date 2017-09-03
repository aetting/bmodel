modelID = '4d'

context_size = 200
retrieval_size = 80

trainingsuf = 'br-orig-ster0.5-dutch'
# trainingsuf = 'br-origfulldutch'
# trainingsuf = 'br-holdoutB-dutch'
#trainingsuf = 'br-nonsterdutch8k'

traincode='full'

maxup = 7000 #7000 in Brouwer
itperup = 100 #100 in Brouwer

embdic='embs/brouwerCOALS-100.txt'
binary = True

tryloc=False
reducelr=False

# dict='brouwerGloVe-100.txt'
# trainingsuf = 'br-origfulleng'

notes='try .5 of ster data to test effect of ratio'
