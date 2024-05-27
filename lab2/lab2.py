import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    images = []
    labels = []

    index = 1
    while index < len(lines):
        if lines[index].strip() == "":
            index += 1
            continue

        image = []
        for _ in range(6):
            while index < len(lines) and lines[index].strip() == "":
                index += 1
            if index >= len(lines):
                break
            row_values = list(map(int, lines[index].strip().split()))
            image.extend(row_values)
            index += 1
        if len(image) == 36:  
            images.append(image)

        while index < len(lines) and lines[index].strip() == "":
            index += 1
        if index >= len(lines):
            break
        label = lines[index].strip()
        if label.isdigit():
            labels.append(int(label))
        index += 1

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    return images, labels

# dataset class
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

# the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(36, 36)
        self.fc2 = nn.Linear(36, 36)
        self.fc3 = nn.Linear(36, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# load data
images, labels = data_from_file('test1.train.txt')

def print_images(images):
    print("Images:")
    for image in images:
        for i in range(0, len(image), 6):
            print(' '.join(map(str, map(int, image[i:i+6]))))
        print()

print_images(images)   
#print("Images:\n", images)
print("Labels:\n", labels)

# creating a dataset of images and labels
dataset = ImageDataset(images, labels)
# creating a data loader for training the model
dataloader = DataLoader(dataset, batch_size=36, shuffle=True)

# initialize the model, loss function, optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print()
# training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_images, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels - 1)
        loss.backward()
        optimizer.step() # оновлення параметрів моделі, використовуючи обчислені градієнти
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

#  the model
model.eval() # модель буде використовуватися лише для прогнозування, а не для навчання.

with torch.no_grad(): # відключаю обчислення градієнтів під час оцінки моделі
    images_tensor = torch.tensor(images)
    labels_tensor = torch.tensor(labels - 1)
    outputs = model(images_tensor)
    _, predicted = torch.max(outputs.data, 1) # обчислює прогнозовані класи 
    #шляхом вибору індексів з найвищими імовірностями для кожного зображення.
    accuracy = (predicted == labels_tensor).sum().item() / len(labels_tensor)
    print(f'Accuracy: {accuracy * 100:.2f}%')
print()

# predictions on the test data
test_images, test_labels = data_from_file('test1.test.txt')


test_dataset = ImageDataset(test_images, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

predicted_classes = []
model.eval()
with torch.no_grad():
    for test_image, _ in test_dataloader:
        output = model(test_image)
        _, predicted_class = torch.max(output.data, 1) # обчислює прогнозовані класи 
        #шляхом вибору індексів з найвищими імовірностями для кожного зображення.
        predicted_classes.append(predicted_class.item() + 1)

print("Передбачені класи:", predicted_classes)

#  predictions
for i, (image, label) in enumerate(zip(test_images, test_labels)):
    train_label = labels[i] if i < len(labels) else 'Unknown'
    print(f"Image {i+1} - Actual: {train_label}, Predicted: {predicted_classes[i]}")

print("Test Images:\n")
print_images(test_images)   
print("Test Labels:\n", test_labels)