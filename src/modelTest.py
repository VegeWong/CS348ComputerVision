import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os

import cDCGAN

# Parameters
image_size = 112
label_dim = 10
G_input_dim = 100
G_output_dim = 3
D_input_dim = 3
D_output_dim = label_dim
num_filters = [1024, 512, 256, 128]

learning_rate = 0.0002
betas = (0.5, 0.999)
batch_size = 12
num_epochs = 20

data_dir = '../../../ILSVRC2012_img_train_par'


data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_datasets = datasets.ImageFolder(data_dir,
                                        transform=data_transforms)

dataloaders = torch.utils.data.DataLoader(dataset=image_datasets,
                                            batch_size=batch_size,
                                            shuffle=True)

# Label list (one-hot)
tmp = torch.LongTensor(range(0, label_dim)).view(-1, 1)
label = torch.zeros(label_dim, label_dim).scatter_(1, tmp, 1).view(-1, label_dim, 1, 1)
fill = torch.zeros([label_dim, label_dim, image_size, image_size]).cuda()
for i in range(label_dim):
    fill[i, i, :, :] = 1

# Models
G = cDCGAN.Generator(G_input_dim, label_dim, num_filters, G_output_dim)

G.load_state_dict(torch.load('./generator_param.pkl'))

for param in G.parameters():
    param.requires_grad = False

for batch_index, (images, labels) in enumerate(dataloaders):
    G_input = torch.randn(batch_size, G_input_dim).view(-1, G_input_dim, 1, 1)
    G_input = G_input.cuda()
    g_images = G(G_input, labels)
    com_images = torch.cat([images, g_images], 1).detach().numpy()
    from PIL import Image
    for i in range(com_images[0]):
        img = Image.fromarray(come_images[i].astype(numpy.uint8))
        img.save(str(i)+'.JPEG')