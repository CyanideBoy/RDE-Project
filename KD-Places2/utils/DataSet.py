from torchvision import datasets
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import numpy as np
import os
import os.path


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

    
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(directory, class_to_idx, extensions=None):
    instances = []
    directory = os.path.expanduser(directory)
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


def pil_loader_img(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def pil_loader_rap(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

    
class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root_img, root_rap, loader_img=pil_loader_img, loader_rap=pil_loader_rap,
                                     extensions=('.jpg','.png'), transform=None):
        super(DatasetFolder, self).__init__(root_img, transform=transform,
                                            target_transform=None)
        classes, class_to_idx = self._find_classes(root_img)
        samples_img = make_dataset(root_img, class_to_idx, extensions)
        samples_rap = make_dataset(root_rap, class_to_idx, extensions)
        if len(samples_img) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(root_img)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader_img = loader_img
        self.loader_rap = loader_rap
        
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples_img = samples_img
        self.samples_rap = samples_rap
        
        self.targets = [s[1] for s in samples_img]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path_img, target_img = self.samples_img[index]
        path_rap, target_rap = self.samples_rap[index]
        sample_img = self.loader_img(path_img)
        sample_rap = self.loader_rap(path_rap)
        if self.transform is not None:
            sample_img = self.transform(sample_img)
            
        paths = (path_img,path_rap)
        return sample_img,sample_rap,target_img,paths


    def __len__(self):
        return len(self.samples_img)