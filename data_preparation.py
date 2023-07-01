import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
import torch

# Load the dataset
df = pd.read_csv("spamdata_v2.csv")

# Check class distribution
class_distribution = df['label'].value_counts(normalize=True)
print("Class Distribution:\n", class_distribution)

# Set random seed for reproducibility
random_seed = 42

# Split the dataset into train, validation, and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'],
                                                                    random_state=random_seed,
                                                                    test_size=0.3,
                                                                    stratify=df['label'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=random_seed,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# Initialize the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length=512,  # Adjust the max length based on the text length distribution
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Convert the tokenized sequences to tensors
train_seq = tokens_train['input_ids']
train_mask = tokens_train['attention_mask']
train_y = torch.tensor(train_labels.tolist())

val_seq = tokens_val['input_ids']
val_mask = tokens_val['attention_mask']
val_y = torch.tensor(val_labels.tolist())

test_seq = tokens_test['input_ids']
test_mask = tokens_test['attention_mask']
test_y = torch.tensor(test_labels.tolist())

# Save the preprocessed data tensors
torch.save(train_seq, 'train_seq.pt')
torch.save(train_mask, 'train_mask.pt')
torch.save(train_y, 'train_y.pt')

torch.save(val_seq, 'val_seq.pt')
torch.save(val_mask, 'val_mask.pt')
torch.save(val_y, 'val_y.pt')

torch.save(test_seq, 'test_seq.pt')
torch.save(test_mask, 'test_mask.pt')
torch.save(test_y, 'test_y.pt')

# Save the tokenizer
tokenizer.save_pretrained('tokenizer')

print("Data preparation completed.")
