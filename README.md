# Surprise Similarity - a context-aware similarity score for vector embeddings

The surprise similarity takes advantage of contextual information to produce a similarity score for embedded objects that more closely mirrors human perception.  Substituting the surprise similarity for common similarity scores (e.g. cosine similarity) has proven to improve results in NLP classification and clustering tasks. Furthermore we use an effective and efficient procedure for fine-tuning sentence-transformer models; we have found this to be valuable in many practical scenarios, e.g. few-shot classification, clustering, and document ranking/retrieval.

## The paper: 
Find an in-depth discussion of surprise similarity definition and results [here](http://arxiv.org).

## Installation 
Get started with:
```
python -m pip install surprise
```

## Use

### Document ranking: a toy example
See surprise_similarity/notebooks/simple_example.ipynb to see details of this example.

In this example we illustrate the difference between the similarity of words as measured by the cosine similarity vs the surprise score.  We rank the words in the vocabulary given by `english_words_alpha_set` (shipped with the package https://pypi.org/project/english-words/) based on their similarity to the word `dog` via cosine similarity (`surprise_weight = 0`) and surprise score (`surprise_weight = 1`) using the following:
```
from english_words import english_words_alpha_set
vocabulary = list(english_words_alpha_set)
dog_cosing_similarity_df = similarity.rank_documents(queries=['dog'],
                                documents=vocabulary,
                                surprise_weight=0,
                                sample_num_cutoff=None,
                                normalize_raw_similarity=False,
                                )

dog_surprise_similarity_df = similarity.rank_documents(queries=['dog'],
                                documents=vocabulary,
                                surprise_weight=1,
                                sample_num_cutoff=None,
                                normalize_raw_similarity=False,
                                )
```                                
Then, if we want to check the rankings of a few examples we can run:
```
my_dogs_name = 'Jude'
example_words = ['the', 'potato', 'my', 'Alsatian', 'furry', 'puppyish', my_dogs_name]
print(Cosine ranks:)
print(dog_cosing_similarity_df[dog_cosing_similarity_df['documents'].isin(example_words)].to_markdown())
print(Surprise ranks:)
print(dog_surprise_similarity_df[dog_surprise_similarity_df['documents'].isin(example_words)].to_markdown())
```
finding:
```
Cosine ranks:
documents      dog
      the 0.852129
   potato 0.850233
       my 0.850161
 Alsatian 0.849615
    furry 0.833391
 puppyish 0.829942
     Jude 0.783955

Surprise ranks:
documents      dog
 Alsatian 0.999325
    furry 0.995553
 puppyish 0.987653
   potato 0.987412
      the 0.979651
       my 0.973795
     Jude 0.639304
```
And finally, we can fine-tune the underlying sentence embedding model to learn my dogs name:
``` 
similarity.train(keys=["dog", 'pet'], queries=[my_dogs_name, my_dogs_name], min_its=30, lr_factor=1)
```
which results in the following cosine-similarity ranking:
```
documents      dog
     Jude 0.929732
   potato 0.929431
       my 0.926398
      the 0.920728
    furry 0.911322
 Alsatian 0.908444
 puppyish 0.904361
 ```
### Classification
Please see the notebook surprise/similarity/notebooks/few_shot_classification.ipynb for full details on how to use the SurpriseSimilarity class to fine-tune and perform a few-shot classification experiment on the `Yahoo! Answers` dataset.

The essentials for fine-tuning are:
```
from surprise_similarity import SurpriseSimilarity
ss = SurpriseSimilarity()
samples = <list of strings to be classified>
labels = <corresponding list of strings to apply to samples>
ss.train(keys=samples, queries=labels)
```
This will fine-tune the classifier, which can then be used for prediction via:
```
test_samples = <list of samples to run inference on>
possible_labels = <list of unique labels that can be assigned>
predictions = ss.predict(keys=test_samples, queries=possible_labels)
```