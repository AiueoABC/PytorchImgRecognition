import torchvision.transforms as transforms
from PIL import Image
import cv2

import torch

"""
Ref: https://ichi.pro/pytorch-no-kenchosei-mappu-o-shiyoshita-nyu-rarunettowa-ku-no-shikakuka-44657159719604
"""

data_path = './datasets/WheelchairNankai/no_exist/160006_844058.jpg'
model_path = './temp/epoch44_loss0.05364989270182217.pth'
label_path = './temp/label.txt'
phase = 'test'

# Read image with PIL
img = Image.open(data_path)

# Load model
model = torch.load(model_path)

# Set dict (Copy from training)
transform_dict = {
    'train': transforms.Compose(
        [transforms.Resize((600, 600)),  # Set image size (Images will be automatically resized)
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
    'test': transforms.Compose(
        [transforms.Resize((600, 600)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}

# Get transformed data
X = transform_dict[phase](img)

# Set model for evaluation
model.eval()

# Get values
scores = model(X.unsqueeze(0).cuda())
score_max_index = scores.argmax()

# Make it to score in range 0-1
scores = torch.nn.Softmax(dim=1)(scores)

# Get labels
with open(label_path) as f:
    labels = f.read().splitlines()

print(f'Image Path: {data_path}\nEstimation: {labels[score_max_index]}, Confidence: {scores[0][score_max_index]}')

image = cv2.imread(data_path)
cv2.putText(image, f'{labels[score_max_index]}, {scores[0][score_max_index]}',
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 125, 255), 3)
cv2.imshow("Annotated", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
