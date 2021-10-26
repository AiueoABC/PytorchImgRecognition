import torchvision.transforms as transforms
from PIL import Image
import cv2

import torch


data_path = './path_to_image.jpg'
model_path = './temp/your_model.pth'
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
         transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3),  # Change colors randomly
         transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # Change sharpness randomly
         transforms.RandomPerspective(),  # random perspective transformation
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         # transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=‘random’, inplace=False),
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

print('\nOther Labels;')
a = sorted([(labels[i], scores[0][i]) for i in range(len(labels))], key=operator.itemgetter(1), reverse=True)
[print(f'{a[i][0]}: {a[i][1]}') for i in range(len(a))]

image = cv2.imread(data_path)
cv2.putText(image, f'{labels[score_max_index]}, {scores[0][score_max_index]}',
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 125, 255), 2)
cv2.imshow("Annotated", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
