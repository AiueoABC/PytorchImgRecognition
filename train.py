import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import os
from random import getrandbits
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision

import torch.nn as nn
import torch

data_folder = './datasets/datasetsroot'
phase = 'train'
train_validation_ratio = 0.9
batch_size = 32
lr = 1e-4
epoch = 100
is_resume = True
if is_resume:
    model_path = './temp/use_your model.pth'
is_fineTune = False  # if True all layers except last layer will be frozen, if false it may consume a lot of memory
is_randomTune = False  # If True all layers are randomly chosen to train model.
use_multi_gpu = False


# To show image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25):
    # Check GPU is available
    use_gpu = torch.cuda.is_available()

    # Start time
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # To save status
    loss_dict ={"train" : [],  "val" : []}
    acc_dict = {"train" : [],  "val" : []}
    
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
        txt2show = f'Epoch {epoch}/{num_epochs - 1}'
        pbar.set_description(f'\n{txt2show}')
        
        # Training/Validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # change to train mode
            else:
                model.eval()  # change to validation mode

            running_loss = 0.0
            running_corrects = 0
            
            iterations_length = len(dataloaders[phase])
            for iteration, data in enumerate(dataloaders[phase]):
                pbar.set_description(f'{txt2show}, {phase}>Processing {iteration}/{iterations_length} ...')
                inputs, labels = data

                # To move data to GPU
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Forward
                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                # torch.max to get value,index
                # torch.max((0.8, 0.1),1)=> (0.8, 0)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            # Calculate loss
            epoch_loss = running_loss / data_size[phase]
            epoch_acc = running_corrects.item() / data_size[phase] if use_gpu else running_corrects / data_size[phase]

            # Save Loss and Accu for later
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            txt2show += f' | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'

            # deep copy the model if found best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  # This is to keep original
                # Save best model
                torch.save(model, f'./temp/epoch{epoch}_loss{epoch_loss}_accu{epoch_acc}.pth')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    print(f'Best val acc: {best_acc:.4f}')

    # best weight and return
    model.load_state_dict(best_model_wts)
    return model, loss_dict, acc_dict


if __name__ == '__main__':
    # Set conditions of transform
    transform_dict = {
        'train': transforms.Compose(
            [transforms.Resize((600, 600)),  # Set image size (Images will be automatically resized)
             transforms.RandomHorizontalFlip(),  # Mirror it randomly
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


    # Load datasets from a specified directory, useful
    data = torchvision.datasets.ImageFolder(root=data_folder, transform=transform_dict[phase])

    # Separate datasets for training and validation
    train_size = int(train_validation_ratio * len(data))
    val_size = len(data) - train_size
    data_size = {"train": train_size, "val": val_size}
    data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])

    # Set DataLoader
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)
    dataloaders = {"train": train_loader, "val": val_loader}

    # Pick images randomly to show samples
    dataiter = iter(dataloaders["train"])
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))  # Show image
    try:
        print(' '.join('%5s' % labels[labels[j]] for j in range(batch_size)))  # Show labels
    except Exception:
        pass

    # Select re-train it or not
    if not is_resume:
        # Load models to use
        # model = models.resnet50(pretrained=True)
        model = models.densenet201(pretrained=True)
        # model = models.vgg16(pretrained=True)
        
        # Rewrite final layer (Maybe adding layer is ok?)
        # Resnet50's final is (fc): Linear(in_features=2048, out_features=1000, bias=True)
        for p in model.parameters():
            p.requires_grad = not is_fineTune if not is_randomTune else getrandbits(1) == 1  # Set False to lock params
        # Change classes number according to actual classes to use
        # By setting this, final layer will be "requires_grad = True"
        # ResNet (fc: final layer of resnet)
        # model.fc = nn.Linear(model.fc.in_features, len(data.class_to_idx))
        # DenseNet
        model.classifier = nn.Linear(model.classifier.in_features, len(data.class_to_idx))
        # VGG (classifier[-1]: final layer of VGG)
        # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(data.class_to_idx))
    else:
        # Load models to use
        model = torch.load(model_path)

        # Lock layers except last layer so to use pretrained model for fine-tuning
        for p in model.parameters():
            p.requires_grad = not is_fineTune if not is_randomTune else getrandbits(1) == 1  # Lock params
        p.requires_grad = True  # Unlock last layer's params

    # Setup model for training
    model = model.cuda() if torch.cuda.is_available() else model
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    if use_multi_gpu and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    # Start training
    model_ft, loss, acc = train_model(model, criterion, optim, num_epochs=epoch)

    # Save final result
    torch.save(model_ft, f'./temp/trained_model_result.pth')
    with open("./temp/label.txt", 'w')  as f:
        f.write('\n'.join(data.classes))

    # To see what happened while training
    loss_train = loss["train"]
    loss_val = loss["val"]
    acc_train = acc["train"]
    acc_val = acc["val"]
    with open('./temp/train_data.dat', 'w') as f:
        f.write('Epoch, TrainLoss, ValLoss, TrainAccu, ValidationAccu\n')
        f.write('\n'.join([f'{i}, {loss["train"][i]}, {loss["val"][i]}, {acc["train"][i]}, {acc["val"][i]}' for i in range(len(loss['train']))]))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # To plot n_rows x cols graph
    # Plot-0
    axes[0].plot(range(epoch), loss_train, label="train")
    axes[0].plot(range(epoch), loss_val, label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    # Plot-1
    axes[1].plot(range(epoch), acc_train, label="train")
    axes[1].plot(range(epoch), acc_val, label="val")
    axes[1].set_title("Acc")
    axes[1].legend()
    # Layout them
    fig.tight_layout()
