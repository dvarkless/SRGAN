import logging
import logging.handlers as log_handlers
from os import listdir
from os.path import join
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from alive_progress import alive_it
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (CenterCrop, Compose, Resize, ToPILImage,
                                    ToTensor)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def get_transformed_pair_plain(hr_img, crop_size):
    hr_img = A.RandomCrop(crop_size, crop_size,
                          always_apply=True)(image=hr_img)
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img['image'])
    lr_img = lr_img['image']
    hr_img = hr_img['image']
    lr_img *= 255  # or any coefficient
    lr_img = lr_img.astype(np.uint8)
    lr_img = ToPILImage()(lr_img)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_transformed_pair_extended(hr_img, crop_size):
    hr_img = A.RandomCrop(crop_size, crop_size, always_apply=True)(
        image=hr_img)
    hr_img = A.RandomBrightnessContrast()(image=hr_img['image'])
    hr_img = A.RandomRotate90()(image=hr_img['image'])
    hr_img = hr_img['image']
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img)
    lr_img = lr_img['image']
    lr_img *= 255  # or any coefficient
    lr_img = lr_img.astype(np.uint8)
    lr_img = ToPILImage()(lr_img)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_transformed_pair_photo(hr_img, crop_size):
    hr_img = A.RandomCrop(crop_size, crop_size,
                          always_apply=True)(image=hr_img)
    hr_img = A.RandomBrightnessContrast()(image=hr_img['image'])
    hr_img = A.RandomRotate90()(image=hr_img['image'])
    hr_img = hr_img['image']
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img)
    lr_img = lr_img['image']
    lr_img *= 255
    lr_img = lr_img.astype(np.uint8)
    lr_img = A.ISONoise(p=0.5)(image=lr_img)
    lr_img = A.JpegCompression(75, 95, p=0.5)(image=lr_img['image'])
    lr_img = lr_img['image']
    lr_img = ToPILImage()(lr_img)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_transformed_pair_game(hr_img, crop_size):
    hr_img = A.RandomCrop(crop_size, crop_size,
                          always_apply=True)(image=hr_img)
    hr_img = A.RandomBrightnessContrast()(image=hr_img['image'])
    hr_img = A.RandomRotate90()(image=hr_img['image'])
    hr_img = A.MotionBlur(blur_limit=(3, 10))(image=hr_img['image'])
    hr_img = hr_img['image']
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img)
    lr_img = lr_img['image']
    lr_img *= 255  # or any coefficient
    lr_img = lr_img.astype(np.uint8)
    lr_img = ToPILImage()(lr_img)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_transformed_pair_video(hr_img, crop_size):
    hr_img = A.RandomCrop(crop_size, crop_size,
                          always_apply=True)(image=hr_img)
    hr_img = A.RandomBrightnessContrast()(image=hr_img['image'])
    hr_img = A.RandomRotate90()(image=hr_img['image'])
    hr_img = A.MotionBlur(blur_limit=(3, 11))(image=hr_img['image'])
    hr_img = hr_img['image']
    # Apply bicubic interpolation:
    lr_img = A.Resize(crop_size // 2, crop_size // 2,
                      interpolation=2, always_apply=True)(image=hr_img)
    lr_img = lr_img['image']
    lr_img *= 255  # or any coefficient
    lr_img = lr_img.astype(np.uint8)
    lr_img = A.ISONoise(p=0.5)(image=lr_img)
    lr_img = A.JpegCompression(75, 95, p=0.5)(image=lr_img['image'])
    lr_img = lr_img['image']
    lr_img = ToPILImage()(lr_img)
    return ToTensor()(lr_img), ToTensor()(hr_img)


def get_logging_handler():
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - [%(module)s] - "%(message)s"')
    logging_handler = log_handlers.TimedRotatingFileHandler(
        'log.log', when='D', interval=2, backupCount=3)
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
    __available_transforms = {
        'plain': get_transformed_pair_plain,
        'extended': get_transformed_pair_extended,
        'photo': get_transformed_pair_photo,
        'game': get_transformed_pair_game,
        'video': get_transformed_pair_video,
    }

    def __init__(self, dataset_dir, crop_size, transform='plain'):
        super().__init__()
        self.image_filenames = [join(dataset_dir, x)
                                for x in listdir(dataset_dir)
                                if is_image_file(x)]
        if crop_size % 2 > 0:
            crop_size -= 1
        self.crop_size = crop_size

        if transform not in self.__available_transforms.keys():
            my_lst = self.__available_transforms.keys()
            msg = f'Choose parameter transform={transform} from {my_lst}'
            raise ValueError(msg)
        self.pair_transform = self.__available_transforms[transform]

    def __getitem__(self, index):
        hr_image = ToTensor()(Image.open(self.image_filenames[index]))
        hr_image = hr_image.cpu().numpy().swapaxes(0, 2)
        lr_image, hr_image = self.pair_transform(
            hr_image,
            self.crop_size)
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
        hr_image = ToTensor()(Image.open(self.image_filenames[index]))
        hr_image = hr_image.cpu().numpy().swapaxes(0, 2)
        w, h = hr_image.shape[0:2]
        crop_size = min(w, h)
        if crop_size % 2 > 0:
            crop_size -= 1
        lr_scale = A.Resize(crop_size // 2, crop_size // 2, interpolation=2)
        hr_scale = A.Resize(crop_size, crop_size, interpolation=2)
        hr_image = A.CenterCrop(crop_size, crop_size)(image=hr_image)['image']
        lr_image = lr_scale(image=hr_image)['image']
        hr_restore_img = hr_scale(image=lr_image)['image']
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
        lr_image = ToTensor()(Image.open(self.lr_filenames[index]))
        lr_image = lr_image.cpu().numpy().swapaxes(0, 2)
        w, h = lr_image.shape[0:2]
        hr_image = ToTensor()(Image.open(self.hr_filenames[index]))
        hr_image = hr_image.cpu().numpy().swapaxes(0, 2)
        hr_scale = A.Resize(2 * h, 2 * w, interpolation=2)
        hr_restore_img = hr_scale(image=lr_image)['image']
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), \
            ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)


def slice_video(video_path, expected_output, output_dir='sliced_data'):
    formats = ['.mp4', '.mp3', '.mkv', '.avi']
    video_path = Path(video_path)
    if not video_path.exists() or video_path.suffix not in formats:
        msg = f'Cannot find video file at path "{video_path}"'
        raise FileNotFoundError(msg)
    capture = cv2.VideoCapture(str(video_path))

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    frame_numbers = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    save_divisor = frame_numbers // expected_output
    assert save_divisor > 0
    progress = alive_it(range(int(frame_numbers)),
                        dual_line=True)
    count = 1
    for i in progress:
        success, frame = capture.read()
        if i == 0:
            progress.title('Slicing video...')
            continue
        if success:
            if i % save_divisor == 0:
                if count > expected_output:
                    break
                if frame.mean() < 5:
                    continue
                img_path = output_dir / f'{count}.png'
                cv2.imwrite(str(img_path), frame)
                count += 1
        progress.text(f'Images so far: ({count})')

    msg = f'Slicing is done, got {count} images'
    print(msg)


def main():
    path = "test_output/upscaled_test_video_480.mp4"
    slice_video(path, 200, output_dir='video/test_dataset')


if __name__ == '__main__':
    main()
