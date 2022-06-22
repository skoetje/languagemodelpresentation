'''
Find changes in topics using the sentence transformers library.
For more info see sbert.net
'''

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# List of sentence from your transcripts
sentences = ['Rijkswaterstaat waarschuwt voor drukte op de weg door boerenprotest',
             'Het aangekondigde protest in Stroe morgen gaat voor flink veel verkeerhsinder zorgen, zegt Rijkswaterstaat',
             'Weggebruikers moeten rekeninghouden met een zware ochtend- en avondspits, maar ook de rest van de dag kan extra druk zijn',
             'Onwaarschijnlijke transferstunt Fortuna: Turkse goalgetter Yilmaz naar Sittard',
            'De Limburgse Club heeft de ervaren Turkse doelpuntenmaker Burak Yilmaz (36) voor vijf jaar vastgelegd.']

#Compute embeddings for the first sentence
first_embedding = model.encode(sentences[0], convert_to_tensor=True)


for i in range(1,len(sentences)):
    # Compute embedding for the current sentence
    embedding = model.encode(sentences[i], convert_to_tensor=True)

    # calculate the cosine similarity of two sentences
    cosine_scores = util.cos_sim(embedding, first_embedding)
    first_embedding = embedding
    print(cosine_scores[0]) # lower means less similar.
