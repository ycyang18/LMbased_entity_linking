
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from src.configs import *
from src.utils import *

#import nltk
#from nltk.tokenize import sent_tokenize, word_tokenize
#nltk.download('punkt')

def load_company_info():
    data_path = os.path.join(FILE, 'company_collection.tsv')
    if not os.path.exists(data_path):
        file_path = os.path.join(FILE, 'company_collection.json')
        all_info = {'name': [], 'url': []}
        file = load_json(file_path)
        for info in file:
            all_info['name'].append(info['name'])
            all_info['url'].append(info['url'])
        all_info = pd.DataFrame.from_dict(all_info)
        all_info.to_csv(data_path, sep='\t', header=True, index=False)
    else:
        all_info = pd.read_csv(data_path, sep='\t', header=0)
    return all_info

def is_company(string) -> bool:
    words = string.split()
    flags = [w[0].isupper() for w in words]
    return False if False in flags else True

# preprocessing
def preprocess_and_check_articles(file_path) -> pd.DataFrame:
    data = []
    company_info = load_company_info()
    all_company_names: List[str] = company_info['name'].tolist()
    all_company_names = [name.lower() for name in all_company_names]
    all_company_names = set(all_company_names)

    i = 0
    with open(file_path, 'r') as file:
        for line in file:
            article: dict = json.loads(line)
            original_text: str = article['text']
            company_names: List[str] = list(article['annotations'].keys()) if 'annotations' in article else []
            company_names: List[str] = [name for name in company_names if name in original_text]
            company_names: List[str] = [name for name in company_names if is_company(name)]
            # Process sentences if all company names are correctly capitalized
            sentences = original_text.split('. ')
            sentences = [sent.strip() for sent in sentences]
            for sentence in sentences:
                data.append({
                    "sentence": sentence,
                    "company_names": [name for name in company_names if name in sentence],
                    "article_order": i
                })
            i += 1
    return pd.DataFrame(data)

# dataset
class ArticleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, max_length: int):
        self.dataframe = dataframe
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def align_labels_with_tokens(self, sentence, company_names):
        tokenized_input = self.tokenizer(
            text=sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_token_type_ids=False,
            return_offsets_mapping=True)
        offset_mappings = tokenized_input['offset_mapping'][0].tolist()
        aligned_labels = [-100] * len(offset_mappings)

        for i, (start, end) in enumerate(offset_mappings):
            if start == end: # Special tokens
                continue
            token = sentence[start: end]
            aligned_labels[i] = 1 if any(token in company for company in company_names) else 0
        name, idxs = [], []
        for i, l in enumerate(aligned_labels):
            if l == -100 or l == 0:
                if len(name) != 0:
                    if " ".join(name) not in company_names:
                        for idx in idxs:
                            aligned_labels[idx] = 0
                    name, idxs = [], []
                continue
            (start, end) = offset_mappings[i]
            token = sentence[start: end]
            name.append(token)
            idxs.append(i)


        return tokenized_input['input_ids'][0], tokenized_input['attention_mask'][0], aligned_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sentence = self.dataframe.iloc[idx]['sentence']
        company_names = self.dataframe.iloc[idx]['company_names']
        input_ids, attention_mask, aligned_labels = self.align_labels_with_tokens(sentence, company_names)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }
