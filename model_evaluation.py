import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast

# Load the preprocessed data tensors
test_seq = torch.load('test_seq.pt')
test_mask = torch.load('test_mask.pt')
test_y = torch.load('test_y.pt')

# Define a batch size
batch_size = 32

# Wrap tensors in a TensorDataset
test_data = TensorDataset(test_seq, test_mask, test_y)

# Create a sequential sampler for test data
test_sampler = SequentialSampler(test_data)

# Create a data loader for test data
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Load the BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Create the BERT model architecture
class BERT_Arch(torch.nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(768, 512)
        self.fc2 = torch.nn.Linear(512, 2)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Create the BERT model instance
model = BERT_Arch(bert)

# Specify the device for evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Load the trained model weights
model.load_state_dict(torch.load('saved_weights.pt'))

# Evaluate the model
model.eval()
total_preds = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        preds = model(sent_id, mask)
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
total_preds = np.concatenate(total_preds, axis=0)
y_pred = np.argmax(total_preds, axis=1)

# Print the classification report
print("Test Classification Report:")
print(classification_report(test_y.numpy(), y_pred))
