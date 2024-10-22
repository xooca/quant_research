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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

# Define the training function
def train_model(model, train_loader, val_loader, optimizer, num_epochs=10, removal_freq=4, removal_fraction=0.1, device='cuda'):
    """
    Train a multi-label model, evaluate after each epoch, and remove confusing data points after every 4 epochs.
    
    :param model: The PyTorch model
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param optimizer: Optimizer (e.g., Adam)
    :param num_epochs: Total number of epochs to train
    :param removal_freq: Number of epochs before removing confusing data
    :param removal_fraction: Fraction of most confusing data points to remove
    :param device: Device to train on ('cuda' or 'cpu')
    """
    
    # Loss function (multi-label: binary cross-entropy with logits)
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # Keep individual losses per data point
    
    model.to(device)

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        individual_losses = []

        # Training phase
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, targets)
            total_batch_loss = loss.mean()
            total_batch_loss.backward()
            optimizer.step()

            # Store total loss and individual losses
            total_loss += total_batch_loss.item()
            individual_losses.append(loss.mean(dim=1).detach().cpu().numpy())

        # Evaluation phase after each epoch
        evaluate_model(model, val_loader, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

        # Every 4 epochs, remove the most confusing data points
        if (epoch + 1) % removal_freq == 0:
            print(f'Removing confusing data points after epoch {epoch + 1}')
            remove_confusing_data(train_loader, individual_losses, removal_fraction)

# Evaluation function (simple accuracy for multi-label classification)
def evaluate_model(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.numel()
    accuracy = total_correct / total_samples
    print(f'Validation Accuracy: {accuracy:.4f}')

# Function to remove the most confusing data points
def remove_confusing_data(train_loader, individual_losses, removal_fraction):
    losses = np.concatenate(individual_losses)
    num_to_remove = int(len(losses) * removal_fraction)
    
    # Identify the data points with the highest loss
    indices_to_remove = np.argsort(losses)[-num_to_remove:]
    
    # Modify the DataLoader to remove those samples
    dataset = train_loader.dataset
    remaining_indices = list(set(range(len(dataset))) - set(indices_to_remove))
    
    # Update the DataLoader with a subset of remaining data points
    new_subset = Subset(dataset, remaining_indices)
    train_loader.dataset = new_subset

    print(f'Removed {num_to_remove} data points, {len(remaining_indices)} remaining.')

# Example of usage:
# model = YourModel()  # Define your model
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# train_loader = DataLoader(your_train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(your_val_dataset, batch_size=32)

# train_model(model, train_loader, val_loader, optimizer)

def remove_learned_data(train_loader, individual_losses, threshold=0.05, consecutive_epochs=3):
    """
    Removes data points that the model has already learned, based on consistently low losses over multiple epochs.
    
    :param train_loader: DataLoader for training data
    :param individual_losses: A list of lists where each sublist corresponds to the losses of the data points for one epoch
    :param threshold: Loss threshold below which data points are considered "learned"
    :param consecutive_epochs: Number of consecutive epochs a data point must have a low loss to be considered learned
    """
    # Stack the losses over multiple epochs to track per-sample loss evolution
    losses_stack = np.stack(individual_losses)  # Shape: (epochs, num_samples)
    
    # Identify data points with losses below the threshold for `consecutive_epochs` epochs
    learned_mask = (losses_stack < threshold).sum(axis=0) >= consecutive_epochs
    
    # Get indices of data points that have not been learned yet
    remaining_indices = np.where(~learned_mask)[0]

    # Modify the DataLoader to keep only the remaining samples
    dataset = train_loader.dataset
    new_subset = Subset(dataset, remaining_indices)
    train_loader.dataset = new_subset

    print(f'Removed {len(learned_mask) - len(remaining_indices)} learned data points, {len(remaining_indices)} remaining.')

# Example usage inside the training loop
def train_model(model, train_loader, val_loader, optimizer, num_epochs=10, removal_freq=4, threshold=0.05, consecutive_epochs=3, device='cuda'):
    """
    Train a multi-label model, evaluate after each epoch, and remove learned data points after every 4 epochs.
    
    :param model: The PyTorch model
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param optimizer: Optimizer (e.g., Adam)
    :param num_epochs: Total number of epochs to train
    :param removal_freq: Number of epochs before removing learned data
    :param threshold: Loss threshold below which data points are considered learned
    :param consecutive_epochs: Number of consecutive epochs a data point must have low loss to be considered learned
    :param device: Device to train on ('cuda' or 'cpu')
    """
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # For multi-label classification
    model.to(device)
    individual_losses_per_epoch = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        individual_losses = []

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, targets)
            total_batch_loss = loss.mean()
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            individual_losses.append(loss.mean(dim=1).detach().cpu().numpy())

        individual_losses_per_epoch.append(np.concatenate(individual_losses))

        # Evaluate after each epoch
        evaluate_model(model, val_loader, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

        # Every `removal_freq` epochs, remove the learned data points
        if (epoch + 1) % removal_freq == 0:
            print(f'Removing learned data points after epoch {epoch + 1}')
            remove_learned_data(train_loader, individual_losses_per_epoch[-removal_freq:], threshold, consecutive_epochs)

# Evaluation function (for multi-label classification)
def evaluate_model(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.numel()
    accuracy = total_correct / total_samples
    print(f'Validation Accuracy: {accuracy:.4f}')



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

from torch.nn.utils.rnn import pad_sequence
import torch

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, batch):
        input_descriptions = [{'input_ids': item['input_ids'], 
                               'attention_mask': item['attention_mask']} 
                              for item in batch]
        class_descriptions = [{'input_ids': item['class_input_ids'], 
                               'attention_mask': item['class_attention_mask']} 
                              for item in batch]
        
        input_batch = self.data_collator(input_descriptions)
        class_batch = self.data_collator(class_descriptions)
        
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_batch['input_ids'],
            'attention_mask': input_batch['attention_mask'],
            'class_input_ids': class_batch['input_ids'],
            'class_attention_mask': class_batch['attention_mask'],
            'labels': labels
        }
