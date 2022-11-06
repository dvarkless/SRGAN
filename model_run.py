from pathlib import Path

import torch

from train import Trainer


def full_test_on_augmentations():
    train_dir = Path('data/compressed_max_full_hd/DIV2K_train_LR_bicubic/')
    eval_dir = Path('data/compressed_max_full_hd/DIV2K_valid_LR_bicubic/')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.Adam
    gen_optimizer_params = {'lr': 5e-4}
    disc_optimizer_params = {'lr': 2e-5}
    aug_lst = ['plain', 'extended', 'photo', 'game', 'video']
    for aug_type in aug_lst:
        trainer = Trainer(crop_size=150, epochs=220, gen_optimizer=gen_optimizer,
                          disc_optimizer=disc_optimizer,
                          gen_optimizer_params=gen_optimizer_params,
                          disc_optimizer_params=disc_optimizer_params,
                          model_type='full', save_interval=25)
        trainer.fit(train_dir, eval_dir, batch_size=16,
                    data_augmentation_type=aug_type,
                    model_tag=aug_type)


def minor_test():
    train_dir = Path('data/compressed_max_full_hd/DIV2K_train_LR_bicubic/')
    eval_dir = Path('data/compressed_max_full_hd/DIV2K_valid_LR_bicubic/')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.Adam
    gen_optimizer_params = {'lr': 5e-4}
    disc_optimizer_params = {'lr': 2e-5}
    trainer = Trainer(crop_size=150, epochs=20, gen_optimizer=gen_optimizer,
                      disc_optimizer=disc_optimizer,
                      gen_optimizer_params=gen_optimizer_params,
                      disc_optimizer_params=disc_optimizer_params)
    trainer.fit(train_dir, eval_dir, batch_size=16,
                data_augmentation_type='plain')


def game_test():
    train_dir = Path('data/no_mans_sky_1080/train')
    eval_dir = Path('data/no_mans_sky_1080/valid')
    gen_optimizer = torch.optim.Adam
    disc_optimizer = torch.optim.AdamW
    gen_optimizer_params = {'lr': 5e-4}
    disc_optimizer_params = {'lr': 2e-5}
    trainer = Trainer(crop_size=150, epochs=400, gen_optimizer=gen_optimizer,
                      disc_optimizer=disc_optimizer,
                      gen_optimizer_params=gen_optimizer_params,
                      disc_optimizer_params=disc_optimizer_params)
    trainer.fit(train_dir, eval_dir, batch_size=20,
                data_augmentation_type='game')


def main():
    game_test()


if __name__ == "__main__":
    main()
