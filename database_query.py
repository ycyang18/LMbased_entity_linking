
#pip install -U --pre "weaviate-client==v4.4b2"
from src.configs import *
from src.utils import *

def create_schema(client, schema=new_schema):
    try:
        client.schema.create(schema)
    except Exception as e:
        print(f"Error creating schema: {e}")

def import_articles(file_path, client):
    with open(file_path, 'r') as file:
        count = 0
        print(f"Importing articles from {file_path} ... ")
        for line in file:
            try:
                article_data = json.loads(line)
                client.data_object.create(
                    data_object={
                        "title": article_data.get("title", ""),
                        "text": article_data.get("text", ""),
                        "source": article_data.get("source", "")
                    },
                    class_name="NewArticles"
                )
                count += 1
            except Exception as e:
                print(f"Error importing article: {e}")
        print(f"--> Total articles imported: {count}")

def import_collections(file_path, client):
    with open(file_path, 'r') as file:
        try:
            company_collections = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            return
    count = 0
    for company_collection_data in tqdm(company_collections, desc="Importing company collection"):
        try:
            client.data_object.create(
                data_object={
                    "name": company_collection_data.get("name", ""),
                    #"founded": company_collection_data.get("founded", ""),
                    #"description": company_collection_data.get("description", ""),
                    "url": company_collection_data.get("url", ""),
                    #"headquarters": company_collection_data.get("headquarters", ""),
                    #"industry_label": company_collection_data.get("industry_label", "")
                },
                class_name="CompanyCollection"
            )
            count += 1
        except Exception as e:
            print(f"Error importing company data: {e}")

    print(f"--> Total companies imported: {count}")

#import_articles(news_new, client)
#import_collections(company_collection, client)

def extract_companies_and_urls_from_database(client):
    article_response = client.query.get("NewArticles", ["title", "text", "source"]).with_limit(60).do()
    articles = article_response['data']['Get']["NewArticles"]
    print(f"Number of articles retrieved: {len(articles)}")
    output_articles = []

    # Extract company names from the articles
    for article in articles:
        truncated_text = article['text'][:4097] #token limits
        print(f"Processing article: {article['title']}")
        prompt = (
            f"Create a dictionary where the keys are the names of companies mentioned in the following article titled '{article['title']}', "
            f"and the values are empty strings. Consider the following article excerpt and its source link for this task. "
            f"Article excerpt: '{truncated_text}'. "
            f"Source link: {article['source']}"
        )
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{
                "role": "user", 
                "content": prompt
            }]
        )

        response_text = chat_response.choices[0].message.content
        try:
            extracted_companies = eval(response_text)
        except SyntaxError:
            extracted_companies = {}

        # Query urls from the collection data using the extracted company names
        annotations = {}
        for company_name in extracted_companies.keys():
            #print(type(company_name))
            # filter
            query_response = client.query.get("CompanyCollection", ["url"]).with_limit(1).with_where({"path": ["name"], "operator": "Equal", "valueString": company_name}).do()
            #query_response = client.query.get("CompanyCollection", ["url"]).with_limit(1).with_where(f"name == '{company_name}'").do()
            company_data = query_response['data']['Get']['CompanyCollection']
            if company_data:
                company_url = company_data[0]['url']
                annotations[company_name] = company_url
            else:
                annotations[company_name] = ''

        output_articles.append({
            "title": article['title'],
            "text": article['text'],
            "annotations": annotations,
            "source": article['source']
        })
    return output_articles

if __name__ == "__main__":
    news_new = os.path.join(FILE, "news_articles-new.jsonl")
    company_collection = os.path.join(FILE, "company_collection.json")
    news_articles_linked = os.path.join(FILE, "news_articles-linked.jsonl")

    # create client & schema
    openai.api_key = OPENAI_API_KEY_FOR_WEAVIATE
    client = initialize_weaviate_client()

    client.schema.delete_class('NewArticles')
    client.schema.delete_class('CompanyCollection')

    client.schema.create(new_schema)


    # import data to weaviate database
    import_articles(news_new, client)
    import_collections(company_collection, client)

    # extract comapny names & url
    output_articles = extract_companies_and_urls_from_database(client)
    with open(news_articles_linked, 'w') as file:
        for article in output_articles:
            json.dump(article, file)
            file.write('\n')
    print(f"\n Linked articles saved to: {news_articles_linked}")



