import logging
import logging.handlers as log_handlers
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import CenterCrop, Compose
from torchvision.transforms import InterpolationMode as Mode
from torchvision.transforms import RandomCrop, Resize, ToPILImage, ToTensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size):
    return Compose([
        ToPILImage(),
        Resize(crop_size // 2, interpolation=Mode.BICUBIC),
        ToTensor()
    ])


def get_logging_handler():
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - [%(module)s] - "%(message)s"')
    logging_handler = log_handlers.TimedRotatingFileHandler(
        'logs/', when='D', interval=2, backupCount=3)
    logging_handler.setFormatter(formatter)
    return logging_handler


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(1080),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super().__init__()
        self.image_filenames = [join(dataset_dir, x)
                                for x in listdir(dataset_dir)
                                if is_image_file(x)]
        if crop_size % 2 > 0:
            crop_size -= 1
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.image_filenames = [join(dataset_dir, x)
                                for x in listdir(dataset_dir)
                                if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = min(w, h)
        if crop_size % 2 > 0:
            crop_size -= 1
        lr_scale = Resize(crop_size // 2, interpolation=Mode.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Mode.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), \
            ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_2/data/'
        self.hr_path = dataset_dir + '/SRF_2/target/'
        self.lr_filenames = [join(self.lr_path, x)
                             for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x)
                             for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((2 * h, 2 * w), interpolation=Mode.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), \
            ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
