from gensim.models import FastText

# Example sentences
sentences = [["I", "love", "machine", "learning"],
             ["FastText", "is", "awesome"]]

# Train FastText model
model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Get word vector for a word
vector = model.wv['machine']
print(vector)