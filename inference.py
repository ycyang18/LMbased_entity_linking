from src.dataset import *
from src.model import *
from src.configs import *

def load_model(model: nn.Module, state_path: str, device=torch.device('cpu')):
    model.load_state_dict(torch.load(state_path, map_location=device))
    return model

@torch.no_grad()
def inference(model: nn.Module, dataloader, device):
    model.eval()
    all_labels, prd_labels, total_loss = [], [], 0
    for batch in tqdm(dataloader, desc=f"Validation"):
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss: torch.Tensor = outputs[0]
        total_loss += loss.item()
        logits: torch.Tensor = outputs[1]
        prd_ids: torch.Tensor = logits.argmax(-1) # (batch_size, max_len)
        all_labels.append(labels.to('cpu'))
        prd_labels.append(prd_ids.to('cpu'))

    all_labels = torch.cat(all_labels, dim=0)
    prd_labels = torch.cat(prd_labels, dim=0)
    avg_loss = total_loss / len(dataloader)
    return prd_labels, all_labels, avg_loss

def eval(prd_labels: torch.Tensor, all_labels: torch.Tensor):
    prd_labels, all_labels = prd_labels.tolist(), all_labels.tolist()
    preds, trues = [], []
    sentence_level_preds, sentence_level_trues = [], []
    assert len(prd_labels) == len(all_labels)
    for i in range(len(all_labels)):
        prd_label = prd_labels[i]
        all_label = all_labels[i]
        prd_label = prd_label[1:-1]
        all_label = all_label[1:-1]
        all_label = [l for l in all_label if l != -100]
        prd_label = prd_label[:len(all_label)]
        preds += prd_label
        trues += all_label
        sentence_level_preds.append(prd_label)
        sentence_level_trues.append(all_label)
    macro_f1 = f1_score(trues, preds, average='macro')
    micro_f1 = f1_score(trues, preds, average='micro')
    accuracy = accuracy_score(trues, preds)
    return {
        'macro_f1': macro_f1, # label imbalance
        'micro_f1': micro_f1, # ignore label imbalance
        'accuracy': accuracy,
    }, sentence_level_preds, sentence_level_trues


## inference: running news_articles-new.jsonl

# Load test data and company info
TEST_PATH = os.path.join(FILE, 'news_articles-new.jsonl')
test_data = preprocess_and_check_articles(TEST_PATH)
company_info = load_company_info()
name2url = {name: url for name, url in zip(company_info['name'], company_info['url'])}

# Prepare model and data loader
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
test_dataset = ArticleDataset(dataframe=test_data, max_length=48)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertEntityLinking(num_labels=2)
model = load_model(model, os.path.join(CHECKPOINT, 'final_model.pt'), device=device)
model.to(device)

# Run inference
prd_labels, all_labels, avg_loss = inference(model, test_loader, device)

# Process predictions and output file
output_articles = []
with open(TEST_PATH, 'r') as f:
    for i, line in enumerate(f):
        article = json.loads(line)
        token_ids, prd_label = test_dataset[i]['input_ids'].to('cpu').tolist(), prd_labels[i].to('cpu').tolist()
        assert len(token_ids) == len(prd_label)

        # Extract company names
        company_names = set()
        idxs = []
        for t, l in zip(token_ids, prd_label):
            if t == 0 or l == 0 and idxs:
                company_names.add(tokenizer.decode(idxs).strip())
                idxs = []
            elif l == 1:
                idxs.append(t)
        annotations = {name: name2url.get(name, "") for name in company_names}
        article['annotations'] = annotations
        output_articles.append(article)

with open(os.path.join(FILE, 'news_articles-linked_bert.jsonl'), 'w') as f:
    for entry in output_articles:
        json.dump(entry, f)
        f.write('\n')