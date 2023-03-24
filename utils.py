import torch
from torchvision import datasets, transforms, utils

import matplotlib.pyplot as plt
import os
import shutil
import splitfolders
import random
from PIL import Image


def get_device():
    """
    :return: GPU device if cuda is available, else CPU device
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def extract_dataset_prepare_folders_split(project_directory, zip_file_name='dataset.zip'):
    """
    Extract all data from the zip file provided and modify folder architecture to match PyTorch ImageFolder requirements
    Split data between training and validation

    :param project_directory: Path to the project directory containing the data zip file and python scripts
    :param zip_file_name: Name of the zip file containing data
    """

    data_path = os.path.join(project_directory, 'dataset')
    zip_path = os.path.join(project_directory, zip_file_name)

    # Unzip data to working directory
    shutil.unpack_archive(os.path.join(project_directory, zip_file_name))

    # Move fields and roads folders into a new one called data_images
    os.makedirs(os.path.join(data_path, 'data_images'))
    shutil.move(os.path.join(data_path, 'fields'),
                os.path.join(data_path, 'data_images'),
                copy_function=shutil.copytree)
    shutil.move(os.path.join(data_path, 'roads'),
                os.path.join(data_path, 'data_images'),
                copy_function=shutil.copytree)
    shutil.rmtree(os.path.join(data_path, 'fields'))
    shutil.rmtree(os.path.join(data_path, 'roads'))

    # Move '3.jpg' and '5.jpg' from data_images/fields into data_images/roads because there are misclassified
    fields_path = os.path.join(data_path, 'data_images', 'fields')
    roads_path = os.path.join(data_path, 'data_images', 'roads')
    shutil.move(os.path.join(fields_path, '3.jpg'),
                os.path.join(roads_path, '3_moved.jpg'))
    shutil.move(os.path.join(fields_path, '5.jpg'),
                os.path.join(roads_path, '5_moved.jpg'))

    # Split the 'data_images' folder into 'train' and 'val' using split-folders
    # Fixed split because the data is imbalanced, validation set will be composed of 10 field and 10 road images
    splitfolders.fixed(os.path.join(data_path, 'data_images'),
                       output=data_path, seed=42, fixed=10, move=True, oversample=False)
    shutil.rmtree(os.path.join(data_path, 'data_images'))

    # Add a parent folder to test images : ./dataset/test_images becomes ./dataset/test/test_images to fit PyTorch ImageFolder requirements
    os.makedirs(os.path.join(data_path, 'test'))
    shutil.move(os.path.join(data_path, 'test_images'),
                os.path.join(data_path, 'test'))


def make_weights_for_balanced_classes(image_labels, n_classes):
    """
    Prepare a weight matrix for a PyTorch WeightedRandomSampler
    The goal is to perform oversampling on classes that are underrepresented on training data
    :param image_labels: List of labels for each image
    :param n_classes: Number of classes (2 in this test)
    :return:
    """

    # Count how many images belong to each classes
    count = [0] * n_classes
    for label in image_labels:
        count[label] += 1

    # The sample weight for each image is equal to (total number of images / number of images belonging to the same class)
    weights = [1/count[label] for label in image_labels]

    return weights


def get_data_loader(data_path, data_split, batch_size, shuffle):
    """
    Returns a data loader for images contained in subfolder of data_folder.
    Subfolders of data_folder correspond to image classes.
    A WeightedRandomSampler is used on training data loader to deal with class imbalance

    :param data_path: path of the folder containing images in it's subfolders
    :param data_split: name of the data split 'train', 'val' or 'test'
    :param batch_size: Batch size of the data loader
    :param shuffle: Boolean, Shuffle the data
    :return: PyTorch DataLoader
    """

    # Get transforms according to data split and create dataset
    transform = get_transform(data_split)
    dataset = datasets.ImageFolder(data_path, transform=transform)

    # if data_split is 'train', create a DataLoader using a WeightedRandomSampler in order to oversample the minority class
    if data_split == "train":
        # Compute sampling weights
        weights = make_weights_for_balanced_classes([dataset.imgs[i][1] for i in range(len(dataset))], len(dataset.classes))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(dataset))
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)

    elif data_split == 'val' or data_split == 'test':
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    else:
        raise ValueError("data_split should be 'train', 'val' or 'test'")

    return data_loader


def get_transform(data_split):
    """
    Returns PyTorch transforms to be applied on images

    :param data_split: Data split name, 'train' or 'test'
    :return: PyTorch transforms according to the data split (train, val or test)
    """

    if data_split == "train":
        # Successive transforms applied to the train images
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(64, 64), scale=(0.5, 1)),
            transforms.ToTensor()
        ])

    elif data_split == "val" or data_split == "test":
        # Successive transforms applied to the test and validation images
        transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor()
        ])

    else:
        raise ValueError("data_split should be 'train', 'val' or 'test'")

    return transform


def normalize(batch_tensor):
    """
    Normalize a batch tensor

    :param batch_tensor: Batch tensor to be normalized
    :return: Normalized batch tensor
    """

    return batch_tensor / torch.linalg.norm(batch_tensor, dim=1, keepdim=True)


def plot_data(x, y_list, labels, x_lab, y_lab, title):
    """
    Plot multiple curves using elements in y_list versus x

    :param x: List containing horizontal coordinates of data points
    :param y_list: List containing lists of vertical coordinates
    :param labels: List containing label for each curve
    :param x_lab: Label of x axis
    :param y_lab: Label of y axis
    :param title: Graph title
    """

    fig = plt.figure()
    # For each list of y coordinates, plot a curve
    for y, label in zip(y_list, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    # plt.ylim(ymin=0)
    plt.title(title)
    plt.legend()
    plt.show()


def show_images_from_data_loader(data_loader):
    """
    Show images from a data loader

    :param data_loader: data loader containing images to be plotted
    """

    images = torch.cat([im for im, _ in data_loader])
    grid = utils.make_grid(images, nrow=5)
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def show_transformation_examples(data_loader):
    """
    Show 5 examples of transformations applied to images before being forwarded through the network

    :param data_loader: Data loader containing training images
    """
    n_image = 5  # Plot 5 examples
    transform = data_loader.dataset.transform

    # Select n_image unique sample from images
    indexes = random.sample(range(len(data_loader.dataset)), n_image)
    img_paths = [data_loader.dataset.imgs[ind][0] for ind in indexes]

    # For each image selected, plot the original image and the transformations output
    _, axs = plt.subplots(n_image, 2, figsize=(10, 10))
    axs[0][0].set_title('Original images')
    axs[0][1].set_title('Transformed images')
    for img_path, ax in zip(img_paths, axs):
        im = Image.open(img_path)
        ax[0].imshow(im)
        ax[0].set_axis_off()

        transformed_im = transform(im)
        ax[1].imshow(transformed_im.permute(1, 2, 0))
        ax[1].set_axis_off()

    plt.show()


def show_image_predictions(data_loader, labels):
    """
    Show images contained in the data loader and their corresponding label predictions

    :param data_loader: Data loader containing images
    :param labels: List of labels sorted in the same way as filenames in data_loader.dataset.imgs (DO NOT SHUFFLE DATA LOADER)
    """

    n_images = len(data_loader.dataset.imgs)

    # Get image paths
    img_paths = [data_loader.dataset.imgs[i][0] for i in range(n_images)]

    # For each image in the data_loader show it and its label prediction as title
    _, axs = plt.subplots(n_images // 2, 2, figsize=(8, 20))
    for img_path, lab, ax in zip(img_paths, labels, axs.flatten()):
        im = Image.open(img_path)
        ax.imshow(im)
        ax.set_title(f'Predicted label : {lab}')
        ax.set_axis_off()

    plt.show()