import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from utils.utils import Config, Progbar, to_cuda, postprocess, stitch_images, imsave
from src.metrics import get_inpainting_metrics
from utils.logger import setup_logger
from torch.utils.data import DataLoader
from src.dataloader_face import FaceDataset
from src.vqgan_models import TSVQGAN
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--config_path', type=str, required=True, help='model config path')
    parser.add_argument('--finetune', default=False, action='store_true', help='whether to local finetune')
    parser.add_argument('--max_iters', type=int, default=150000, required=False,
                        help='max train steps, train 150k, finetune 300k')
    parser.add_argument('--learning_rate', type=float, default=2e-4, required=False,
                        help='learning rate, train 2e-4, finetune 4e-5')
    parser.add_argument('--gpu', type=str, required=True, help='gpu ids')

    args = parser.parse_args()
    args.path = os.path.join('check_points', args.path)
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile(args.config_path, config_path)

    # load config file
    config = Config(config_path)
    config.path = args.path
    config.gpu_ids = args.gpu
    config.d_lr = args.learning_rate
    config.g_lr = args.learning_rate
    if not args.finetune: # combined only used in finetuning
        config.combined = False

    log_file = 'log-{}.txt'.format(time.time())
    logger = setup_logger(os.path.join(args.path, 'logs'), logfile_name=log_file)
    for k in config._dict:
        logger.info("{}:{}".format(k, config._dict[k]))

    # save samples and eval pictures
    os.makedirs(os.path.join(args.path, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(args.path, 'eval'), exist_ok=True)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids

    # init device
    if torch.cuda.is_available():
        config.device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.device = torch.device("cpu")
    n_gpu = torch.cuda.device_count()

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)

    # load dataset
    train_list = config.data_flist[config.dataset]['train']
    val_list = config.data_flist[config.dataset]['val']
    eval_path = config.data_flist[config.dataset]['test']
    if args.finetune: # load mask for finetuning
        fixed_mask_path = config.data_flist[config.dataset]['test_mask']
        irr_path = config.irr_path
        seg_path = config.seg_path
    else:
        irr_path = None
        seg_path = None
        fixed_mask_path = None
    train_dataset = FaceDataset(config, train_list, irr_mask_path=irr_path,
                                seg_mask_path=seg_path, training=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    val_dataset = FaceDataset(config, val_list, fix_mask_path=fixed_mask_path, training=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=2,
        drop_last=False,
        shuffle=False,
        collate_fn=train_dataset.collate_fn
    )
    sample_iterator = val_dataset.create_iterator(config.sample_size)
    model = TSVQGAN(config, logger=logger)

    if args.finetune and config.restore is False:
        finetune_g_path = model.g_path + '_last.pth'
        finetune_d_path = model.d_path + '_last.pth'
        model.load_for_finetune(finetune_g_path, finetune_d_path)
    else:
        model.load(is_test=False)

    steps_per_epoch = len(train_dataset) // config.batch_size
    iteration = model.iteration
    epoch = model.iteration // steps_per_epoch
    logger.info('Start from epoch:{}, iteration:{}'.format(epoch, iteration))

    model.train()
    keep_training = True
    best_score = {}
    while (keep_training):
        epoch += 1

        stateful_metrics = ['epoch', 'iter', 'g_lr']
        progbar = Progbar(len(train_dataset), max_iters=steps_per_epoch,
                          width=20, stateful_metrics=stateful_metrics)
        for items in train_loader:
            model.train()
            items = to_cuda(items, config.device)
            _, g_loss, d_loss, logs = model.get_losses(items)
            model.backward(g_loss=g_loss, d_loss=d_loss)
            iteration = model.iteration

            logs = [("epoch", epoch), ("iter", iteration), ('g_lr', model.g_sche.get_lr()[0])] + logs
            progbar.add(config.batch_size, values=logs)

            if iteration % config.log_iters == 0:
                logger.debug(str(logs))

            if iteration % config.sample_iters == 0:
                model.eval()
                with torch.no_grad():
                    items = next(sample_iterator)
                    items = to_cuda(items, config.device)
                    fake_img = model(items['img'], items['mask'])
                    show_results = [postprocess(fake_img)]
                    if args.finetune:
                        images = stitch_images(postprocess(items['img'] * (1 - items['mask']) + items['mask']),
                                               show_results, img_per_row=2)
                    else:
                        images = stitch_images(postprocess(items['img']), show_results, img_per_row=2)
                sample_name = os.path.join(args.path, 'samples', str(iteration).zfill(7) + ".png")

                print('\nsaving sample {}\n'.format(sample_name))
                images.save(sample_name)

            if iteration % config.eval_iters == 0:
                model.eval()
                eval_progbar = Progbar(len(val_dataset), width=20)
                index = 0
                with torch.no_grad():
                    for items in val_loader:
                        items = to_cuda(items, config.device)
                        fake_img = model(items['img'], items['mask'])
                        fake_img = postprocess(fake_img)  # [b, h, w, 3]
                        for i in range(fake_img.shape[0]):
                            sample_name = os.path.join(args.path, 'eval',
                                                       val_dataset.load_name(index)).replace('.jpg', '.png')
                            imsave(fake_img[i], sample_name)
                            index += 1

                        eval_progbar.add(fake_img.shape[0])

                score_dict = get_inpainting_metrics(eval_path, os.path.join(args.path, 'eval'),
                                                    logger, fid_test=config.fid_test)
                if config.save_best and 'fid' in score_dict:
                    if 'fid' not in best_score or best_score['fid'] >= score_dict['fid']:
                        best_score = score_dict.copy()
                        best_score['iteration'] = iteration
                        model.save(prefix='best_fid')

            if iteration % config.save_iters == 0:
                model.save(prefix='last')

            if iteration >= args.max_iters:
                keep_training = False
                break

    logger.info('Best score: ' + str(best_score))
