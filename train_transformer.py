import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from utils.utils import Config, Progbar, to_cuda, postprocess, stitch_images
from utils.logger import setup_logger
from torch.utils.data import DataLoader
from src.dataloader_face import FaceDataset
from src.transformer_models import GETransformer
import time
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='model checkpoints path')
    parser.add_argument('--gpu', type=str, required=True, help='gpu ids')
    parser.add_argument('--sketch_model_path', type=str, required=True, help='Sketch vqvae weights')
    parser.add_argument('--image_model_path', type=str, required=True, help='Image vqvae weights')
    parser.add_argument('--config_path', type=str, required=True, help='model config path')
    parser.add_argument('--max_iters', type=int, default=300000, required=False,
                        help='max train steps, transformer 300k')
    parser.add_argument('--learning_rate', type=float, default=5e-5, required=False, help='learning rate')

    args = parser.parse_args()
    args.path = os.path.join('check_points', args.path)
    config_path = os.path.join(args.path, 'transformer_config.yml')

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
    config.lr = args.learning_rate
    config.max_iters = args.max_iters

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
    sketch_train_list = config.data_flist[config.dataset]['train_cond']
    sketch_val_list = config.data_flist[config.dataset]['val_cond']
    eval_path = config.data_flist[config.dataset]['test']
    fixed_mask_path = config.data_flist[config.dataset]['test_mask']
    irr_path = config.irr_path
    seg_path = config.seg_path
    train_dataset = FaceDataset(config, train_list, skflist=sketch_train_list,
                                irr_mask_path=irr_path, seg_mask_path=seg_path, training=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=8,
        drop_last=True,
        shuffle=True
    )
    val_dataset = FaceDataset(config, val_list, skflist=sketch_val_list,
                              fix_mask_path=fixed_mask_path, training=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=2,
        drop_last=False,
        shuffle=False
    )
    sample_iterator = val_dataset.create_iterator(config.sample_size)
    model = GETransformer(config, args.sketch_model_path, args.image_model_path, logger=logger)

    model.load(is_test=False)
    model.restore_from_stage1()
    steps_per_epoch = len(train_dataset) // config.batch_size
    iteration = model.iteration
    epoch = model.iteration // steps_per_epoch
    logger.info('Start from epoch:{}, iteration:{}'.format(epoch, iteration))

    model.train()
    keep_training = True
    best_score = {}
    # make some AR samples randomly
    if config.lm_rate > 0:
        lm_size = int(config.batch_size * config.lm_rate)
        mc = [1] * lm_size + [0] * (config.batch_size - lm_size)
    else:
        mc = None
    while (keep_training):
        epoch += 1

        stateful_metrics = ['epoch', 'iter', 'lr']
        progbar = Progbar(len(train_dataset), max_iters=steps_per_epoch,
                          width=20, stateful_metrics=stateful_metrics)
        for items in train_loader:
            model.train()
            items = to_cuda(items, config.device)
            if config.lm_rate > 0:
                random.shuffle(mc)
                items['mc'] = torch.tensor(mc).to(config.device)
            else:
                items['mc'] = None
            # if lm_rate=1 means that all targets are masked (AR)
            if config.lm_rate == 1:
                items['mask'] = torch.ones_like(items['mask'])
            loss, logs = model.get_losses(items)
            model.backward(loss)
            iteration = model.iteration

            logs = [("epoch", epoch), ("iter", iteration), ('lr', model.sche.get_lr()[0])] + logs
            progbar.add(config.batch_size, values=logs)

            if iteration % config.log_iters == 0:
                logger.debug(str(logs))

            if iteration % config.sample_iters == 0:
                model.eval()
                with torch.no_grad():
                    items = next(sample_iterator)
                    items = to_cuda(items, config.device)
                    # if lm_rate=1 means that all targets are masked (AR)
                    if config.lm_rate == 1:
                        items['mask'] = torch.ones_like(items['mask'])
                    fake_imgs = []
                    fake_imgs_sampled = []
                    for i in tqdm(range(items['img'].shape[0])):
                        fake_img = model.sample(items['img'][i:i + 1],
                                                items['sketch'][i:i + 1],
                                                items['mask'][i:i + 1],
                                                temperature=config.temperature,
                                                greed=True, top_k=None)
                        fake_img_sampled = model.sample(items['img'][i:i + 1],
                                                        items['sketch'][i:i + 1],
                                                        items['mask'][i:i + 1],
                                                        temperature=config.temperature,
                                                        greed=False, top_k=config.sample_topk)
                        fake_imgs.append(fake_img)
                        fake_imgs_sampled.append(fake_img_sampled)
                    fake_imgs = torch.cat(fake_imgs, dim=0)
                    fake_imgs_sampled = torch.cat(fake_imgs_sampled, dim=0)
                    combined_imgs = items['img'] * (1 - items['mask']) + \
                                    items['sketch'].repeat(1, 3, 1, 1) * items['mask']
                    show_results = [postprocess(combined_imgs),
                                    postprocess(fake_imgs),
                                    postprocess(fake_imgs_sampled)]
                    images = stitch_images(postprocess(items['img']), show_results, img_per_row=1)
                sample_name = os.path.join(args.path, 'samples', str(iteration).zfill(7) + ".png")

                print('\nsaving sample {}\n'.format(sample_name))
                images.save(sample_name)

            if iteration % config.eval_iters == 0:
                model.eval()
                eval_progbar = Progbar(len(val_dataset), width=20)
                index = 0
                # testing perplexity (AR inference is too slow)
                ppls = []
                with torch.no_grad():
                    for items in val_loader:
                        items = to_cuda(items, config.device)
                        # if lm_rate=1 means that all targets are masked (AR)
                        if config.lm_rate == 1:
                            items['mask'] = torch.ones_like(items['mask'])
                        ppl = model.perplexity(items['img'], items['sketch'], items['mask'])
                        ppls.append(ppl)
                        eval_progbar.add(items['img'].shape[0])

                mean_ppl = np.mean(ppls)
                print('PPL:{}'.format(mean_ppl))
                logger.info('PPL:{}'.format(mean_ppl))
                if config.save_best:
                    if 'ppl' not in best_score or mean_ppl <= best_score['ppl']:
                        best_score['ppl'] = mean_ppl
                        best_score['iteration'] = iteration
                        model.save(prefix='best_ppl')

            if iteration % config.save_iters == 0:
                model.save(prefix='last')

            if iteration >= config.max_iters:
                keep_training = False
                break

    logger.info('Best score: ' + str(best_score))
