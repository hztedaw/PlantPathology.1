# ## Kaggle - Plant Pathology Challenge
# ### Task to be solved:
# to develop a model to classify apple leaves as healthy or infected with specific diseases.
# ##Link to the challenge: 
# https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7
# ### Background: 
# The goal of this challenge is to develop models that can accurately classify images of apple leaves as either healthy or infected with one of several diseases. This can aid in early detection and treatment of plant diseases, ultimately improving apple production in the end.
# ### Data: 
# The dataset consists of images of apple leaves, with each image labeled according to the disease that has infected the leaf, or "healthy" if the leaf is not infected. The total size of the dataset is 823.79 MB. In addition to the image data, there is one academic paper that is directly readable and could provide valuable context or insights for the analysis.     
# ### Strategy: 
# 
# A deep learning approach, specifically using a Convolutional Neural Network (CNN), will be employed to classify the images based on the labels provided. Could be doing the following or more or not:                                             
# 1.	Data Preprocessing: load the dataset and split it into training, validation, and test sets. Normalize the pixel values of the images to be between 0 and 1, as this can help in speeding up the training; augment the data to increase the size of the training dataset, which can help in improving the model's performance.
# 
# 2.	Model Building and Transfer Learning: choose a pre-trained model that is suitable for the task based on the literature search. Implement feature extraction or fine-tuning as per the requirement of the task. This involves removing or adding a layer and specifying the loss function, optimizer, and evaluation metrics.
# 
# 3.	Training: train the CNN on the training dataset while also validating it on the validation dataset. The model will learn to classify the images based on the disease label.  Monitor the training process and adjust the model architecture or hyperparameters as necessary to improve performance.                                                  
# 
# 4.	Evaluation and Optimization: once the model is trained, evaluate its performance on the test dataset to understand how well it generalizes to new, unseen data. with fine-tune the model and make any necessary adjustments to improve its performance. This could include changing the model architecture, adjusting hyperparameters, or using more sophisticated data augmentation techniques.
# 

# ### Import all libraries
import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.optim as optim

print(torch.cuda.is_available())
torch.cuda.set_device(0)

# # Data Processing

# ## Build dataset
# 
# ### We build our dataset using pytorch Dataset class. We split the dataset into train and test set with a ratio of 8:2

class PlantDataset(Dataset):

    def __init__(self, root, phase="train", transformation=None):
        
        self.image_root = os.path.join(root, "images")
        self.transformation = transformation
        with open(os.path.join(root, "train.csv"), newline='') as f:
            self.infos = [i for i in csv.reader(f, delimiter=',')]
            self.infos.pop(0)

        image_paths = [os.path.join(self.image_root, f"{i[0]}.jpg") for i in self.infos]
        labels = torch.tensor([int(i[1]) for i in self.infos])

        train_number = int(len(labels) * 0.9)

        # total_number = int(len(labels))
        # train_number = int(total_number * 0.7)
        # val_number = int(total_number * 0.15)


        # elif phase=="val":
        #     self.image_paths = image_paths[train_number: train_number+val_number]
        #     self.labels = labels[train_number: train_number+val_number]
        # elif phase=="test":
        #     self.image_paths = image_paths[train_number+val_number:]
        #     self.labels = labels[train_number+val_number:]

        if phase=="train":
            self.image_paths = image_paths[:train_number]
            self.labels = labels[:train_number]
        elif phase=="test":
            self.image_paths = image_paths[train_number:]
            self.labels = labels[train_number:]
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        label = self.labels[item]

        if self.transformation is not None:
            image = self.transformation(image)
        return image, label

plant_dataset = PlantDataset("data", "train")


# ## Visualization
# ### Here are some visualization of samples from the dataset. The title "1" indicates the leave is healthy.

# def showImagesHorizontallyWithLabels(pil_imgs: list):
#     fig = figure(figsize=(20, 12))
#     img_num = len(pil_imgs)
#     for ind, (image, label) in enumerate(pil_imgs):
#         a = fig.add_subplot(1, img_num, ind+1)
#         a.title.set_text(label.item())
#         imshow(np.asarray(image))
#         axis('off')

# showImagesHorizontallyWithLabels([plant_dataset[i] for i in range(5)])

# quick check for img size, to see if loaded correctly  
print(f"The shape of the images in the dataset is {plant_dataset[0][0].size}")

# image data normalization, effectively scales the pixel values to be in the range [-1, 1]
train_transformation = transforms.v2.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.v2.RandomHorizontalFlip(p=0.5),
    transforms.v2.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

test_transformation = transforms.v2.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.v2.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

train_dataloader = DataLoader(PlantDataset(root="data", phase="train", transformation=train_transformation), batch_size=32, shuffle=True)
test_dataloader = DataLoader(PlantDataset(root="data", phase="test", transformation=test_transformation), batch_size=32, shuffle=False)

data_batch, label_batch = next(iter(train_dataloader))
print(f"data_batch: {data_batch.shape} label_batch: {label_batch.shape}")


# # Building model and training code

# ### Build a function that includes both training and testing code.

def train(model, criterion, train_dataloader, test_dataloader, epoch=1):

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epoch):  # loop over the dataset multiple times
        
        train_iter = tqdm(train_dataloader)

        loss_total = 0.0

        model = model.cuda()
        for (inputs, labels) in train_iter:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_iter.set_description(f"Loss: {loss.item()}")
            loss_total += loss.item()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {epoch} AVG Loss: {loss_total/len(train_dataloader)} lr: {current_lr}")

    print('Finished training. Start testing.')

    correct_pred, total_pred = 0, 0
    with torch.no_grad():
        for (inputs, labels)  in test_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred += 1
                total_pred += 1
    print(f"Model accuracy: {correct_pred/total_pred*100:.1f}%")


# ### Load a resnet50 pretrained with ImageNet

resnet50 = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
resnet50.fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=128, bias=True),
    nn.Linear(in_features=128, out_features=32, bias=True),
    nn.Linear(in_features=32, out_features=2, bias=True),
)
resnet50.cuda()


# ### Train the model

criterion = nn.CrossEntropyLoss()
train(resnet50, criterion, train_dataloader, test_dataloader, epoch=10)





