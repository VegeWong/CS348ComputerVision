import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os

import cDCGAN

# Parameters
image_size = 112
label_dim = 1000
G_input_dim = 1000
G_output_dim = 3
D_input_dim = 3
D_output_dim = 1
num_filters = [1024, 512, 256, 128]

learning_rate = 0.0002
betas = (0.5, 0.999)
batch_size = 128
num_epochs = 20

data_dir = '/run/media/vegw/新加卷/ImageNet/train'


data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_datasets = datasets.ImageFolder(data_dir,
                                        transform=data_transforms)

dataloaders = torch.utils.data.DataLoader(dataset=image_datasets,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)

# Label list (one-hot)
tmp = torch.LongTensor(range(0, label_dim)).view(-1, 1)
label = torch.zeros(label_dim, label_dim).scatter_(1, tmp, 1).view(-1, G_input_dim, 1, 1)

# Models
G = cDCGAN.Generator(G_input_dim, label_dim, num_filters, G_output_dim)
D = cDCGAN.Discriminator(D_input_dim, label_dim, num_filters[::-1], D_output_dim)
# G.cuda()
# D.cuda()

# Loss function
criterion = torch.nn.BCELoss()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=betas)


for batch_index, (images, labels) in enumerate(dataloaders):
    
    batch_size = images.size()[0]
    real_lab = label[labels]

    real_img = images
    # real_img = images.cuda()
    # real_lab = labels.cuda()

    # Train descriminator with real data
    D_real_decision = D(real_img, real_lab)
    D_real_loss = criterion(D_real_decision, real_lab)

    # Train discriminator with fake data
    G_input = torch.randn(batch_size, G_input_dim).view(-1, G_input_dim, 1, 1)
    # G_input = G_input.cuda()
    fake_label = label[(torch.rand(batch_size) * label_dim).type(torch.LongTensor)]
    fake_img = G(G_input, fake_label)
    # fake_label = fake_label.cuda()
    # fake_img = fake_img.cuda()

    D_fake_decision = D(fake_img, fake_label)
    D_fake_loss = criterion(D_fake_decision, fake_label)

    # Back propagation
    D_loss = D_real_loss + D_fake_loss
    D,zero_grad()
    D_loss.backward()
    D_optimizer.step()

    # Train generator
    G_input = torch.randn(batch_size, G_input_dim).view(-1, G_input_dim)
    # G_input = G_input.cuda()
    fake_label = label[(torch.rand(batch_size) * label_dim).type(torch.LongTensor)]
    fake_img = G(G_input, fake_label)
    # fake_label = fake_label.cuda()
    # fake_img = fake_img.cuda()
        
    D_fake_decision = D(fake_img, fake_label)
    G_loss = criterion(D_fake_decision, fake_label)

    # Back propagation
    G.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    # loss values
    # D_losses.append(D_loss.data[0])
    # G_losses.append(G_loss.data[0])

    print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
            % (epoch+1, num_epochs, i+1, len(data_loader), D_loss.data[0], G_loss.data[0]))
    
    break