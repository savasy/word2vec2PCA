

from gensim.models import Word2Vec

TRAIN=[["the","apple","is","red"], ["we","are","not","so","powerfull"],]
# put your TRAIN data
token_count = sum([len(s) for s in TRAIN])

model = Word2Vec(size=300, min_count=5, workers=8, window=5)
model.build_vocab(TRAIN)
model.train(TRAIN,total_examples=len(TRAIN), epochs=model.iter)

with open( "tensors.csv", 'w+') as tensors:
    with open( "meta.csv", 'w+') as metadata:
         for word in model.wv.index2word:
           metadata.write(str(word) + '\n')
           vector_row = '\t'.join(map(str, model[word]))
           tensors.write(vector_row + '\n')



