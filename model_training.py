import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast
import numpy as np

# Load the preprocessed data tensors
train_seq = torch.load('train_seq.pt')
train_mask = torch.load('train_mask.pt')
train_y = torch.load('train_y.pt')

val_seq = torch.load('val_seq.pt')
val_mask = torch.load('val_mask.pt')
val_y = torch.load('val_y.pt')

test_seq = torch.load('test_seq.pt')
test_mask = torch.load('test_mask.pt')
test_y = torch.load('test_y.pt')

# Define a batch size
batch_size = 16  # Reduce the batch size to save memory

# Wrap tensors in a TensorDataset
train_data = TensorDataset(train_seq, train_mask, train_y)

# Create a random sampler for training data
train_sampler = RandomSampler(train_data)

# Create a data loader for training data
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Wrap tensors in a TensorDataset
val_data = TensorDataset(val_seq, val_mask, val_y)

# Create a sequential sampler for validation data
val_sampler = SequentialSampler(val_data)

# Create a data loader for validation data
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Define the BERT architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Create the BERT model architecture
model = BERT_Arch(bert)

# Specify the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# Define the loss function
cross_entropy = nn.NLLLoss()

# Define early stopping variables
early_stopping_counter = 0
early_stopping_patience = 5
best_val_loss = float('inf')

# Define the number of training epochs
epochs = 50

# Train the model
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_preds = []
    for step, batch in enumerate(train_dataloader):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_train_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    y_train_pred = np.argmax(total_preds, axis=1)

    model.eval()
    total_loss = 0
    total_preds = []
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            batch = [t.to(device) for t in batch]
            sent_id, mask, labels = batch
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
        avg_val_loss = total_loss / len(val_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
        y_val_pred = np.argmax(total_preds, axis=1)

    # Print metrics
    print(f"Epoch {epoch+1}/{epochs}")
    print("Train Loss:", avg_train_loss)
    print("Validation Loss:", avg_val_loss)
    print("Train Classification Report:")
    print(classification_report(train_y.numpy(), y_train_pred))
    print("Validation Classification Report:")
    print(classification_report(val_y.numpy(), y_val_pred))

    # Update learning rate scheduler
    scheduler.step(avg_val_loss)

    # Check early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        # Save the best model weights
        save_path = 'saved_weights.pt'
        torch.save(model.state_dict(), save_path)
        print("Saved model updated.")
    else:
        early_stopping_counter += 1

    # Check if early stopping criteria are met
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered. No improvement in validation loss for", early_stopping_patience, "epochs.")
        break

print("Model training completed.")
