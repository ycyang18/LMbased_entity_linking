
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split

from src.dataset import *
from src.model import *
from src.configs import *
from inference import *

# train
def train(augmented_data_path, batch_size, learning_rate, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertEntityLinking(num_labels=2)
    model.to(device)
    print(f'Preparing dataset...')
    df = preprocess_and_check_articles(augmented_data_path)
    # print("-> All company names capitalized correctly in articles:", all_companies_correct)

    dataset = ArticleDataset(dataframe=df, max_length=48)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, no_deprecation_warning=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}: [Train Loss = {avg_train_loss:.6f}]")

        # inference 
        prd_labels, all_labels, avg_val_loss = inference(model, val_loader, device)
        val_acc = eval(prd_labels, all_labels)[0]['accuracy']
        print(f"Epoch {epoch + 1}/{epochs}: [Val Loss = {avg_val_loss:.6f}, Val Acc = {val_acc:.6f}]")
        torch.save(model.state_dict(), os.path.join(CHECKPOINT, f'model_epoch_{epoch + 1}.pt'))
    torch.save(model.state_dict(), os.path.join(CHECKPOINT, 'final_model.pt'))

aug_gold_path = os.path.join(FILE, f"augmented_8000_news_articles-gold.jsonl")
train(augmented_data_path=aug_gold_path, batch_size=32, learning_rate=2e-5, epochs=3)