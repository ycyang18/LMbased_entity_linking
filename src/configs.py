import os
import sys

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC   = os.path.join(ROOT, 'src')
FILE = os.path.join(ROOT, 'file')
CHECKPOINT   = os.path.join(ROOT, 'checkpoint')
sys.path.append(ROOT)

OPENAI_API_KEY = 'place_your_API_key'
OPENAI_API_KEY_FOR_WEAVIATE = 'place_your_API_key'

new_schema = {
    "vectorizer": "text2vec-openai", 
    "classes": [
        {
            "class": "NewArticles",
            "vectorizer": "text2vec-openai",
            "vectorizeClassName": True,
            "description": "New article dataset",
            "properties": [
                {
                    "name": "title",
                    "description": "The title of the news article",
                    "dataType": ["string"] #text?
                },
                {
                    "name": "text",
                    "description": "The full-text article as extracted from the news website",
                    "dataType": ["string"]
                },
                {
                    "name": "source",
                    "description": "The link to the original article",
                    "dataType": ["string"],
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                }
            ],
            "moduleConfig": {
                "text2vec-openai": {},
                "generative-openai": {
                    "model": "gpt-3.5-turbo-16k",
                    "maxTokensProperty": 8192
                }
            }
        }
    ]
}


collection_schema = {
    "vectorizer": "text2vec-openai",
    "classes": [
        {
            "class": "CompanyCollection",
            "vectorizer": "text2vec-openai",
            "vectorizeClassName": True,
            "description": "A collection of companies",
            "properties": [
                {
                    "name": "name",
                    "description": "The name of the company",
                    "dataType": ["string"]
                },
                {
                    "name": "founded",
                    "description": "The year the company was founded",
                    "dataType": ["string"]
                },
                {
                    "name": "description",
                    "description": "A short text describing the company",
                    "dataType": ["string"]
                },
                {
                    "name": "url",
                    "description": "URL of the company website, which is a unique identifier",
                    "dataType": ["string"]
                },
                {
                    "name": "headquarters",
                    "description": "The location of the company (country, state, city)",
                    "dataType": ["string"]
                },
                {
                    "name": "industry_label",
                    "description": "One or multiple industry labels assigned to this company",
                    "dataType": ["string"]
                }
            ],
            "moduleConfig": {
                "text2vec-openai": {},
                "generative-openai": {
                    "model": "gpt-3.5-turbo-16k",
                    "maxTokensProperty": 8192
                }
            }
        }
    ]
}

gold_schema = {
    "vectorizer": "text2vec-openai",
    "classes": [
        {
            "class": "AnnotationItem",
            "description": "Represents an annotation item with a company name and URL",
            "properties": [
                {
                    "name": "companyName",
                    "description": "The name of the company mentioned in the article",
                    "dataType": ["text"]
                },
                {
                    "name": "companyURL",
                    "description": "The URL corresponding to the mentioned company",
                    "dataType": ["text"]
                }
            ]
        },
        {
            "class": "Articles",
            "vectorizer": "text2vec-openai",
            "vectorizeClassName": True,
            "description": "Augmented article dataset",
            "properties": [
                {
                    "name": "title",
                    "description": "The title of the news article",
                    "dataType": ["text"]
                },
                {
                    "name": "text",
                    "description": "The full-text article as extracted from the news website",
                    "dataType": ["text"]
                },
                {
                    "name": "annotations",
                    "description": "A list of references to AnnotationItems that appear in the article",
                    "dataType": ["AnnotationItem"]
                },
                {
                    "name": "source",
                    "description": "The link to the original article",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                }
            ],
            "moduleConfig": {
                "text2vec-openai": {},
                "generative-openai": {
                    "model": "gpt-3.5-turbo-16k",
                    "maxTokensProperty": 8192
                }
            }
        }
    ]
}