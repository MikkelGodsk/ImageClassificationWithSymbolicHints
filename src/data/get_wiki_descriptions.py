import wikipediaapi as wiki
from wikidata.client import Client
import sys
import json
import os
from tqdm import tqdm

directory = ""  #Insert directory where mapping.json is located
fname = os.path.join(directory, 'mapping.json')

with open(fname, 'r') as f_obj:
    mappings = json.load(f_obj)

for k, v in mappings.items():
    mappings[k] = v.split('/')[-1]
    


client = Client()
wiki_en = wiki.Wikipedia('en')

articles = {}
failures = []

# Cache in case we would like to modify the policy and not have to make new queries.
cache_wikidata = {}
cache_wikipedia = {}

use_cache = True

i = 0
for k, v in tqdm(mappings.items()):
    i+=1
    entity = None
    if v not in cache_wikidata.keys() or not use_cache:
        entity = client.get(v, load=True)
        cache_wikidata[v] = entity
    else:
        entity = cache_wikidata[v]
    title = None
    
    # First look up Wikipedia page in Wikidata. Else search Wikipedia for its title. Else count as a failure
    if 'enwiki' in entity.data['sitelinks'].keys():
        title = entity.data['sitelinks']['enwiki']['title']
    elif 'en' in entity.data['labels']:
        title = entity.data['labels']['en']['value']
    else:
        failures.append(k)
        continue
        
    if title not in cache_wikipedia.keys() or not use_cache:
        article = wiki_en.page(title)
        cache_wikipedia[title] = article
    else:
        article = cache_wikipedia[title]
    if not article.exists():
        failures.append(k)
    else:
        articles[k] = article
    
print('\n')
print("Number of failures: {}".format(len(failures)))
print("Failures:")
print(failures)



import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')

mappings_reversed = {v:k for k, v in mappings.items()}

get_synset_from_id = lambda synset_id: wn.synset_from_pos_and_offset(synset_id[0], int(synset_id[1:]))
get_title_from_synset = lambda synset: synset.name().split('.')[0].replace('_', ' ')

print("Failures:")
for failure in failures:
    print('WordNet id: {} - Title: {}'.format(failure, get_title_from_synset(get_synset_from_id(failure))))


print("\n\nDifference in titles:")
for wnid, article in articles.items():
    wordnet_title = get_title_from_synset(get_synset_from_id(wnid)).lower()
    article_title = article.title.lower()
    if wordnet_title not in article_title and wordnet_title not in article.summary.lower():
        print("WordNet id: {} - WordNet title: {} - Wikipedia title: {}".format(wnid, wordnet_title, article_title))

summaries = {}
length = len(articles)
i=0
for k, v in tqdm(articles.items()):
    i+=1
    summaries[k] = v.summary#.replace('\n', ' ')
    
with open('wiki_descriptions.json', 'w') as f_obj:
    json.dump(summaries, f_obj)