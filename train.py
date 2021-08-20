import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import os
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

    for epoch in tqdm(range(num_epochs)):
        print('\n')
        if (epoch+1)%5 == 0:  # Show current epoch once in 5 times
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Training/Validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                # To move data to GPU
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                #~~~~~~~~~~~~~~forward~~~~~~~~~~~~~~~
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

            # Save
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model if found best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  # This is to keep original
                torch.save(model, f'./temp/epoch{epoch}_loss{epoch_loss}_accu{epoch_acc}.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:.4f}'.format(best_acc))

    # best weight and return
    model.load_state_dict(best_model_wts)
    return model, loss_dict, acc_dict


if __name__ == '__main__':
    # Set conditions of transform
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

    # Load datasets from a specified directory
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
    print(' '.join('%5s' % labels[labels[j]] for j in range(8)))  # Show labels

    # Load models to use
    model = models.resnet50(pretrained=True)

    # Rewrite final layer (Maybe adding layer is ok?)
    # Resnet50's final is (fc): Linear(in_features=2048, out_features=1000, bias=True)
    for p in model.parameters():
        p.requires_grad = False  # Lock params
    model.fc = nn.Linear(2048, 2)  # Rewrite the number on right is len(data.class_to_idx)

    # Setup model for training
    model = model.cuda() if torch.cuda.is_available() else model
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Start training
    model_ft, loss, acc = train_model(model, criterion, optim, num_epochs=epoch)
    
    # Save final result
    torch.save(model_ft, f'./temp/trained_model_result.pth')
    with open("label.txt", 'w')  as f:
        f.write('\n'.join(data.classes))

    # To see what happened while training
    loss_train = loss["train"]
    loss_val = loss["val"]
    acc_train = acc["train"]
    acc_val = acc["val"]
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
