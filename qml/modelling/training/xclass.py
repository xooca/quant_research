from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

class XMLDataset(Dataset):
    def __init__(self, input_descriptions, class_descriptions, labels, tokenizer, max_len=128):
        self.input_descriptions = input_descriptions
        self.class_descriptions = class_descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input_descriptions)

    def __getitem__(self, idx):
        # Tokenize the input description
        input_encodings = self.tokenizer(self.input_descriptions[idx], 
                                         truncation=True, padding='max_length', 
                                         max_length=self.max_len, return_tensors='pt')
        # Tokenize the class description
        class_encodings = self.tokenizer(self.class_descriptions[idx], 
                                         truncation=True, padding='max_length', 
                                         max_length=self.max_len, return_tensors='pt')
        # Get the labels
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'class_input_ids': class_encodings['input_ids'].squeeze(),
            'class_attention_mask': class_encodings['attention_mask'].squeeze(),
            'labels': labels
        }

# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare DataLoader
def create_dataloader(input_descriptions, class_descriptions, labels, tokenizer, batch_size=16, max_len=128):
    dataset = XMLDataset(input_descriptions, class_descriptions, labels, tokenizer, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage:
train_dataloader = create_dataloader(
    input_descriptions=descriptions,  # List of input descriptions
    class_descriptions=class_descriptions,  # List of class descriptions
    labels=y_train,  # Multi-hot encoded labels
    tokenizer=tokenizer,  # Hugging Face BERT tokenizer
    batch_size=16
)

import torch
import torch.nn as nn
from transformers import BertModel

class DeepXMLWithClassDescriptions(nn.Module):
    def __init__(self, num_classes, hidden_size=768):
        super(DeepXMLWithClassDescriptions, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # Pre-trained BERT for both input and class descriptions
        self.hidden_size = hidden_size
        
        # A fully connected layer for classification
        self.fc = nn.Linear(2 * hidden_size, num_classes)  # Concatenation of both input and class embeddings
    
    def forward(self, input_ids, attention_mask, class_input_ids, class_attention_mask):
        # Get BERT embeddings for the input description
        input_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        input_embedding = input_outputs.pooler_output  # (batch_size, hidden_size)
        
        # Get BERT embeddings for the class description
        class_outputs = self.bert(input_ids=class_input_ids, attention_mask=class_attention_mask)
        class_embedding = class_outputs.pooler_output  # (batch_size, hidden_size)
        
        # Concatenate input and class embeddings
        combined_embedding = torch.cat([input_embedding, class_embedding], dim=1)  # (batch_size, 2 * hidden_size)
        
        # Pass through a fully connected layer for multi-label classification
        logits = self.fc(combined_embedding)  # (batch_size, num_classes)
        
        return torch.sigmoid(logits)  # Sigmoid for multi-label classification

# Define model
num_classes = len(class_descriptions)  # Set this according to the number of unique classes
model = DeepXMLWithClassDescriptions(num_classes)

from transformers import AdamW
from torch.optim import AdamW

# Binary Cross-Entropy Loss for multi-label classification
criterion = nn.BCELoss()

# AdamW optimizer (from Hugging Face)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
def train_model(model, train_dataloader, num_epochs=3):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            class_input_ids = batch['class_input_ids']
            class_attention_mask = batch['class_attention_mask']
            labels = batch['labels']

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                            class_input_ids=class_input_ids, class_attention_mask=class_attention_mask)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}")

# Train the model
train_model(model, train_dataloader, num_epochs=3)
\from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            class_input_ids = batch['class_input_ids']
            class_attention_mask = batch['class_attention_mask']
            labels = batch['labels']
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                            class_input_ids=class_input_ids, class_attention_mask=class_attention_mask)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    
    # Binary predictions (threshold at 0.5)
    binary_preds = (preds > 0.5).astype(int)
    
    # Compute evaluation metrics
    precision = precision_score(labels, binary_preds, average='micro')
    recall = recall_score(labels, binary_preds, average='micro')
    f1 = f1_score(labels, binary_preds, average='micro')
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Evaluate the model on the validation or test set
evaluate_model(model, train_dataloader)
# Prediction function
def predict(model, dataloader, threshold=0.5):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            class_input_ids = batch['class_input_ids']
            class_attention_mask = batch['class_attention_mask']
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                            class_input_ids=class_input_ids, class_attention_mask=class_attention_mask)
            
            all_preds.append(outputs.cpu().numpy())
    
    preds = np.vstack(all_preds)
    # Apply threshold to get binary predictions
    binary_preds = (preds > threshold).astype(int)
    
    return binary_preds

# Predict on new data
predictions = predict(model, train_dataloader)

# Example input descriptions for prediction
new_input_descriptions = [
    "This course focuses on building machine learning models using Python libraries such as Scikit-Learn and TensorFlow.",
    "This workshop covers software development best practices, including version control, continuous integration, and unit testing."
]

# Example class descriptions (10 different classes)
new_class_descriptions = [
    "Python for data analysis",
    "Introduction to TensorFlow",
    "Advanced deep learning techniques",
    "Data structures and algorithms in Python",
    "Software engineering practices",
    "Machine learning fundamentals",
    "Object-oriented programming in Java",
    "Web development with JavaScript",
    "Data visualization with Matplotlib",
    "Version control with Git"
]

# Just for testing, we assume labels are known for these inputs (in practice, no labels during prediction)
# Labels (multi-label, for reference)
# Input 1 should be tagged to class 1, 2, and 6 (Machine Learning-related)
# Input 2 should be tagged to class 5 and 10 (Software Engineering-related)
new_labels = [
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # Input 1 tags
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]   # Input 2 tags
]
# Create DataLoader for the new prediction data
predict_dataloader = create_dataloader(
    input_descriptions=new_input_descriptions,
    class_descriptions=new_class_descriptions, 
    labels=new_labels,  # Dummy labels, not used in prediction
    tokenizer=tokenizer,
    batch_size=2  # Since we only have 2 inputs here
)
