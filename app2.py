from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW

# Example dataset (replace with your actual dataset)
X_train = ["Description of food item 1", "Description of food item 2", ...]
y_train = [[1, 0, 1, 0, ...], [0, 1, 0, 1, ...], ...]  # Binary labels for each category

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize the input descriptions
encoded_data_train = tokenizer.batch_encode_plus(
    X_train,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    X_val,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

# Create TensorDatasets for training and validation
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(y_train)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(y_val)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# Define DataLoader for batching and shuffling
batch_size = 32
dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_val = DataLoader(dataset_val, 
                            sampler=SequentialSampler(dataset_val), 
                            batch_size=batch_size)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(y_train[0]),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(),
                  lr=2e-5, 
                  eps=1e-8)

# Fine-tune BERT for sequence classification
epochs = 4

for epoch in range(epochs):
    model.train()
    for batch in dataloader_train:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       
        outputs = model(**inputs)  # Forward pass
        loss = outputs.loss
        loss.backward()            # Backward pass
        optimizer.step()
        optimizer.zero_grad()
        
    # Validation
    model.eval()
    for batch in dataloader_val:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }
        with torch.no_grad():
            outputs = model(**inputs)  # Forward pass
            
# Example of predicting categories for a new food item
new_food_item = ["Description of the new food item"]
encoded_new_food_item = tokenizer.batch_encode_plus(
    new_food_item,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)
input_ids = encoded_new_food_item['input_ids']
attention_mask = encoded_new_food_item['attention_mask']
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_categories = logits.cpu().numpy()

print("Predicted categories:", predicted_categories)
