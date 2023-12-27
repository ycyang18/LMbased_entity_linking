# LM-based Entity Linking 

This project is the realization of Entity Linking task using two apporahes: Vector Database Integration with OpenAI's Generative Model and Fine-tuning BERT for Token Classification.

## Installation

Clone the repository and install the dependencies:
```
git clone https://github.com/ycyang18/LM-based-EntityLinking.git
cd LM-based-EntityLinking
pip install -r requirements.txt
```

## Methods

### Approach 1: Vector Database Integration with OpenAI's Generative Model
This approach uses Embedded Weaviate configued with generative model `gpt-3.5-turbo` to extract the company names and urls through the power of RAG. 

1. Data Preparation: Create schemas (can be found under `src/configs.py`) and import `company_collection.json` and `news_articles-new.jsonl` into the local vector database.
2. Entity Extraction: Utilize Weaviate's Generative Search (ChatCompletion) to identify company names within articles.
3. URL Retrieval: Perform database queries to associate identified company names with their respective URLs.

#### Implementation:
Add your API keys to `src/configs.py`:
```python
OPENAI_API_KEY = 'place_your_API_key'
OPENAI_API_KEY_FOR_WEAVIATE = 'place_your_API_key'
```
```
python database_query.py
```


### Apporach 2: Fine-tuning BERT for Token Classification

1. Data Augmentation: Generate 8000 fictional articles (JSON objects) with identical format as `news_articles-gold.jsonl`, using OpenAI's (`gpt-3.5-turbo` & `gpt-4.0`) new JSON mode feature. The augmented dataset is stored in `file/augmented_8000_news_articles-gold.jsonl`.
2. Model Training: Fine-tune the BERT based model for token classification tasks on the augmented dataset. BERT cased is used to remain the capitalized tokens to increase the sensitivity of the model on company name detection.

#### Implementation:
```
python src/data_augmentation.py
python train.py
python src/inference.py
```

## Result
You can find output using two approaches in the following:
1. Approach1: `file/news_articles-linked.jsonl`
2. Approach2: `file/news_articles-linked_bert.jsonl`
