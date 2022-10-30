from pathlib import Path
from math import log10

import logging
from datetime import date
import torch
import pandas as pd
import torch.optim as optim
import torch.utils.data as tdata
import torchvision.utils as utils
from torch.autograd import Variable
from torch.autograd.anomaly_mode import set_detect_anomaly
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, get_logging_handler
from loss import GeneratorLoss
from model import Generator, Discriminator

from alive_progress import alive_it


class Trainer:
    def __init__(self, crop_size=120, epochs=100, 
                 gen_optimizer=None, disc_optimizer=None, 
                 gen_optimizer_params=None, disc_optimizer_params=None, 
                 verbose_logs=False, gen_model_name=None, 
                 disc_model_name=None) -> None:

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose_logs else logging.INFO)
        self.logger.addHandler(get_logging_handler())
        self.logger.info('======= TRAINER STARTED ======')
        set_detect_anomaly(True)
        self.cuda = torch.cuda.is_available()
        if not self.cuda:
            self.logger.error(f'Could not detect GPU on board, aborting (torch.cuda.is_available={self.cuda})')
            raise OSError('Could not detect GPU on board, aborting')

        if gen_model_name and disc_model_name:
            self.logger.info(f'Loading models from files "{gen_model_name}" and "{disc_model_name}"')
            self.gen_model = torch.load(gen_model_name)
            self.disc_model = torch.load(disc_model_name)
        else:
            self.logger.info('Initializing new models')
            self.gen_model = Generator()
            self.disc_model = Discriminator()

        gen_optimizer_params = gen_optimizer_params if gen_optimizer_params else dict()
        disc_optimizer_params = disc_optimizer_params if disc_optimizer_params else dict()
        gen_optimizer_params['params'] = self.gen_model.parameters()
        disc_optimizer_params['params'] = self.disc_model.parameters()

        gen_optimizer = gen_optimizer if gen_optimizer else optim.Adam 
        disc_optimizer = disc_optimizer if disc_optimizer else optim.Adam 
        self.gen_optimizer = gen_optimizer(**gen_optimizer_params)
        self.disc_optimizer = disc_optimizer(**disc_optimizer_params)

        self.gen_criterion = GeneratorLoss()

        self.epochs = epochs
        self.crop_size = crop_size

        creation_time_str = date.today().isoformat()
        self.out_path = Path(f'training_results/SRF_{creation_time_str}')
        self.logger.debug(f'writing data into "{self.out_path}"')
        if not self.out_path.exists():
            self.logger.debug(f'Creating directory "{self.out_path}"')
            self.out_path.mkdir()

        self.trainer_results = []

        self.gen_model.cuda()
        self.disc_model.cuda()
        self.gen_criterion.cuda()

    def fit(self, train_hr_dir, eval_hr_dir, batch_size=32):
        train_hr_dir = Path(train_hr_dir)
        eval_hr_dir = Path(eval_hr_dir)
        train_set = TrainDatasetFromFolder(train_hr_dir, crop_size=self.crop_size)
        eval_set = ValDatasetFromFolder(eval_hr_dir)

        train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(dataset=eval_set, num_workers=4, batch_size=1, shuffle=False)
        
        bar = alive_it(range(self.epochs), self.epochs)
        for epoch in bar:
            gen_loss, disc_loss = self._train_epoch(train_loader)
            print(f'============= {epoch+1}/{self.epochs} =============')
            print(f'Training losses: Generator: {gen_loss:.4f}, Discriminator: {disc_loss:.4f}')
            results = self._eval_epoch(eval_loader, epoch)
            print(f'Evaluating model:')
            for name, val in results.items():
                print(f'{name} = {val:.4f}')

            results['gen_loss'] = gen_loss
            results['disc_loss'] = disc_loss
            self.trainer_results.append(results.copy())
            del results


    def _train_epoch(self, loader):
        if not loader:
            raise ValueError('DataLoader is not defined')
        self.gen_model.train()
        self.disc_model.train()
        self.gen_criterion.clear_losses()

        for data, target, in loader:
            hr_img = Variable(target).cuda()
            lr_img = Variable(data).cuda()

            sr_img = self.gen_model(lr_img)

            self.disc_model.zero_grad()

            real_out = self.disc_model(hr_img).mean()
            fake_out = self.disc_model(sr_img).mean()
            disc_loss = 1 - real_out + fake_out
            disc_loss.backward()
            self.disc_optimizer.step()

            self.gen_model.zero_grad()
            sr_img = self.gen_model(lr_img)
            fake_out = self.disc_model(sr_img).mean()
            gen_loss = self.gen_criterion(fake_out, sr_img, hr_img)
            gen_loss.backward()
            self.gen_optimizer.step()

        print('Singular losses:')
        img_l, adv_l, perc_l = self.gen_criterion.get_losses()
        print(f' Image loss = {img_l:.3f}; Adversarial loss = {adv_l:.3f}; Perception loss = {perc_l:.3f}')
        return gen_loss.item(), disc_loss.item()

    def _eval_epoch(self, loader, epoch):
        self.gen_model.eval()

        with torch.no_grad():
            evaling_results = {'mse': 0, 'psnr': 0, 'ssim': 0, 'batch_size': 0}
            eval_images = []
            for eval_lr, eval_restored, eval_hr in loader:
                eval_lr = eval_lr.cuda()
                eval_hr = eval_hr.cuda()
                batch_size = eval_lr.size(0)
                evaling_results['batch_size'] += batch_size

                eval_sr = self.gen_model(eval_lr)
                
                batch_mse = torch.nn.functional.mse_loss(eval_sr, eval_hr)
                evaling_results['mse'] += batch_mse * batch_size

                batch_ssim = pytorch_ssim.ssim(eval_sr, eval_hr).item()
                evaling_results['psnr'] = 10 * log10((eval_hr.max()**2) / (evaling_results['mse'] / evaling_results['batch_size']))
                evaling_results['ssim'] = batch_ssim / evaling_results['batch_size']
                eval_images.extend(
                        [display_transform()(eval_restored.squeeze(0)), display_transform()(eval_hr.data.cpu().squeeze(0)),
                         display_transform()(eval_sr.data.cpu().squeeze(0))])
            eval_images = torch.stack(eval_images)
            eval_images = torch.chunk(eval_images, eval_images.size(0) // 15)
            index = 1
            for image in eval_images:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, self.out_path / f'epoch_{epoch}_N{index}_psnr-{evaling_results["psnr"]:.3f}db_ssim-{evaling_results["ssim"]:.3f}.png', padding=5)
                index += 1
                if index > 5:
                    break

        return evaling_results

    def get_metric_list(self, metric_name):
        out_data = []
        if not (metric_name in self.trainer_results[0].keys()):
            msg = f'Wrong metric name: "{metric_name}"'
            self.logger.error(msg)
            print(msg)
            raise KeyError(msg)

        for item in self.trainer_results:
            item = item.cpu() if isinstance(item, torch.Tensor) else item
            out_data.append(item[metric_name])

        return out_data

if __name__ == "__main__":
    train_dir = Path('data/compressed_max_640_480/DIV2K_train_HR/')
    eval_dir = Path('data/compressed_max_640_480/DIV2K_valid_HR/')
    gen_optimizer = torch.optim.AdamW
    disc_optimizer = torch.optim.AdamW
    gen_optimizer_params = {'lr': 1e-3}
    disc_optimizer_params = {'lr': 2e-3}
    trainer = Trainer(crop_size=200, epochs=50, gen_optimizer=gen_optimizer, 
                      disc_optimizer=disc_optimizer, gen_optimizer_params=gen_optimizer_params,
                      disc_optimizer_params=disc_optimizer_params)
    trainer.fit(train_dir, eval_dir, batch_size=12)
    plt.plot(trainer.get_metric_list('mse'))
    plt.plot(trainer.get_metric_list('ssim'))
    plt.show()
    print('===============================')
    plt.plot(trainer.get_metric_list('gen_loss'))
    plt.plot(trainer.get_metric_list('disc_loss'))
    plt.show()
