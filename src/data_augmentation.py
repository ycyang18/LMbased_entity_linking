import os
import json
import random
import numpy as np
import pandas as pd
import openai
import time
from tqdm import tqdm

from configs import *

def preprocessing_gold(gold_path):
    all_data, url_list, total_annotations = {}, [], 0
    with open(gold_path, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            all_data[i] = data
            annotations_values = list(data['annotations'].values())
            url_list.extend(annotations_values)
            total_annotations += len(data['annotations'])
    avg_annotation_length = total_annotations / len(all_data)
    return all_data, url_list, avg_annotation_length

def load_gold_articles(file_path):
    articles = []
    with open(file_path, 'r') as file:
        for line in file:
            articles.append(json.loads(line))
    return articles

def generate_articles(sample_articles, example_urls, num_to_generate=1, max_retries=5):
    generated_articles = []
    word_counts = [len(article['text'].split()) for article in sample_articles]

    for _ in tqdm(range(num_to_generate), desc="Generating articles"):
        target_word_count = random.choice(word_counts)
        style_example = random.choice(sample_articles)['text'][:500]
        url_example = random.choice(example_urls)
        num_annotations = random.randint(1, 10)

        prompt_content = (
            "Create a fictional news article in JSON format with exactly the following structure. "
            "The article should have approximately {0} words and {1} annotations."
            "The article writing style should resemble this provided example: '{2}'. "
            "Each annotation includes imaginative company names as key and the corresponding URLs as value. "
            "The URLs are in the format like example URLs: '{3}'."
            "Ensure that company names are capitalized and spelled identically in the article and annotations. "
            "Here's the JSON Object template:".format(target_word_count, num_annotations, style_example, url_example)
        ) + '''
            {
                "title": "<article title>",
                "text": "<full-text of the article>",
                "annotations": {
                    "<Company Name 1>": "<company1.com>",
                    "<Company Name 2>": "<company2.com>",
                    ...
                },
                "source": "<link to the original article>"
            }
        '''
        for attempt in range(max_retries):
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt_content}]
                )
                try:
                    article = json.loads(completion['choices'][0]['message']['content'])
                    if all(key in article for key in ['title', 'text', 'annotations', 'source']):
                        generated_articles.append(article)
                        break
                    else:
                        print("Generated article missing fields, regenerating...")
                except json.decoder.JSONDecodeError:
                    print("JSONDecodeError encountered, skipping this article.")
            except openai.error.ServiceUnavailableError:
                print("Service unavailable, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(5)
            except openai.error.Timeout as e:
                print(f"Timeout error: {e}, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(5)

    return generated_articles

def generate_articles_batch(sample_articles, example_urls, batch_size, max_retries=5):
    generated_articles = []
    word_counts = [len(article['text'].split()) for article in sample_articles]

    for _ in range(batch_size):
        target_word_count = random.choice(word_counts)
        style_example = random.choice(sample_articles)['text'][:500]
        url_example = random.choice(example_urls)
        num_annotations = random.randint(1, 10)

        prompt_content = (
            "Create a fictional news article in JSON format with exactly the following structure. "
            "The article should have approximately {0} words and {1} annotations."
            "The article writing style should resemble this provided example: '{2}'. "
            "Each annotation includes imaginative company names as key and the corresponding URLs as value. "
            "The URLs are in the format like example URLs: '{3}'."
            "Ensure that company names are capitalized and spelled identically in the article and annotations. "
            "Here's the JSON Object template:".format(target_word_count, num_annotations, style_example, url_example)
        ) + '''
            {
                "title": "<article title>",
                "text": "<full-text of the article>",
                "annotations": {
                    "<Company Name 1>": "<company1.com>",
                    "<Company Name 2>": "<company2.com>",
                    ...
                },
                "source": "<link to the original article>"
            }
        '''

        for attempt in range(max_retries):
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt_content}]
                )
                try:
                    article = json.loads(completion['choices'][0]['message']['content'])
                    if all(key in article for key in ['title', 'text', 'annotations', 'source']):
                        generated_articles.append(article)
                        break
                    else:
                        print("Generated article missing fields, regenerating...")
                except json.decoder.JSONDecodeError:
                    print("JSONDecodeError encountered, skipping this article.")
            except (openai.error.ServiceUnavailableError, openai.error.Timeout) as e:
                print(f"Error: {e}, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(5)

    return generated_articles

def save_articles(file_path, articles, mode='a'):
    with open(file_path, mode) as f:
        for article in articles:
            f.write(json.dumps(article) + '\n')

if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY
    news_gold = os.path.join(FILE, "news_articles-gold.jsonl")
    gold_articles = load_gold_articles(news_gold)
    
    num_to_generate = 8000
    batch_size = 1000
    file_path = os.path.join(FILE, f"augmented_{num_to_generate}_news_articles-gold.jsonl")

    for batch in tqdm(range(0, num_to_generate, batch_size), desc="Processing batches"):
        generated_articles = generate_articles_batch(gold_articles, preprocessing_gold(news_gold)[1], batch_size=batch_size)
        save_articles(file_path, generated_articles, mode='a' if batch > 0 else 'w')

    print(f"Completed processing {num_to_generate} articles. Saved to {file_path}")