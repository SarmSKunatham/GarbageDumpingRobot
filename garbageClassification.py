import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import warnings
import cv2
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)


model_path = "GarbageClassificationModelWeight.pth"

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

load_model = ResNet()
load_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
load_model.eval()
print("Model loaded !!!")

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
device = get_default_device()
print("Device:", device)
datasetClasses = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'] 

'''There will be 3 main categories of garbage:
    1. Paper (cardboard + paper)
    2. Bottle (Plastic + Glass)
    3. Unknow (Metal + Trash)
'''
mainClass = ['paper', 'bottle', 'unknown']
mainCategory = {
    0: 0,
    1: 1,
    2: 2,
    3: 0,
    4: 1,
    5: 2
}


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    print(f"Prob: {prob} and Preds: {preds}")

    if prob[0].item() < 0.8:
        return mainClass[2]

    mainCategoryOutput = mainCategory[preds[0].item()]

    # Retrieve the class label
    # return datasetClasses[preds[0].item()]
    return mainClass[mainCategoryOutput]

def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    plt.xticks([]), plt.yticks([])
    label = predict_image(example_image, load_model)
    plt.title(label)
    plt.show()
    # destroy all the windows that are currently open.
    cv2.destroyAllWindows()
    print("The image resembles", label + ".")

# predict_external_image("test1.jpg")
# predict_external_image("test2.jpg")
# predict_external_image("cardboard9.jpg")
# predict_external_image("glass7.jpg")

frameWidth = 1080
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

while cap.isOpened():
    success, img = cap.read()
    if success:
        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # If pressed 's' key, then capture the image and save it
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('test.jpg', img)
            predict_external_image("test.jpg")
            # After predicting the image, delete the image
            os.remove("test.jpg")