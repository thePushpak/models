import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, BertTokenizerFast

# Load the BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

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

# Create an instance of the BERT_Arch class with the loaded BERT model
model = BERT_Arch(bert)

# Load the saved model weights
model.load_state_dict(torch.load('saved_weights.pt'))

# Specify the device for testing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Preprocess the real data
text = "Get rich quick! Invest in this amazing opportunity now and become a millionaire!"
encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

# Pass the real data through the model to obtain predictions
with torch.no_grad():
    preds = model(input_ids, attention_mask)

# Convert the predictions to class labels
threshold = 0.5
predicted_class = 'spam' if preds[0][1] > threshold else 'not spam'

# Perform any necessary post-processing on the predictions

# Print the model's prediction
print("Prediction:", predicted_class)
