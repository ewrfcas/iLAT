import torch
import torch.nn as nn
import os
import functools
import torch.optim as optim
from src.layers import Decoder
from src.plm_layers import Encoder
from src.quantize import VectorQuantizer
from src.loss import PerceptualLoss, VGG19, AdversarialLoss
from utils.utils import torch_show_all_params, get_lr_schedule_with_steps, torch_init_model, get_combined_mask
import torch.nn.functional as F

try:
    from apex import amp

    amp.register_float_function(torch, 'matmul')
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class TSVQGAN(nn.Module):
    def __init__(self, config, logger=None):
        super(TSVQGAN, self).__init__()
        self.config = config
        self.iteration = 0
        self.name = config.model_type
        self.g_path = os.path.join(config.path, self.name + '_g')
        self.d_path = os.path.join(config.path, self.name + '_d')
        loss_config = config.model['params']['lossconfig']['params']
        self.eps = 1e-6

        if config.combined:
            self.g_model = VQCombineModel(config).to(config.device)
        else:
            self.g_model = VQModel(config).to(config.device)
        self.d_model = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(config.device)
        if logger is not None:
            logger.info('Generator Parameters:{}'.format(torch_show_all_params(self.g_model)))
            logger.info('Discriminator Parameters:{}'.format(torch_show_all_params(self.d_model)))
        else:
            print('Generator Parameters:{}'.format(torch_show_all_params(self.g_model)))
            print('Discriminator Parameters:{}'.format(torch_show_all_params(self.d_model)))

        # loss
        self.perceptual_weight = config.perceptual_weight
        self.codebook_weight = loss_config['codebook_weight']
        self.mask_cb_weight = loss_config['mask_cb_weight']
        self.valid_cb_weight = loss_config['valid_cb_weight']
        self.disc_factor = loss_config['disc_factor']
        self.discriminator_weight = loss_config['disc_weight']
        self.discriminator_iter_start = loss_config['disc_start']
        vgg = VGG19(pretrained=True, vgg_norm=config.vgg_norm).to(config.device)
        vgg.eval()
        self.perceptual_loss = PerceptualLoss(vgg, weights=config.vgg_weights,
                                              reduction='mean').to(config.device)
        self.adversarial_loss = AdversarialLoss(type=loss_config['gan_type']).to(config.device)

        self.g_opt = optim.Adam(params=self.g_model.parameters(),
                                lr=float(config.g_lr), betas=(config.beta1, config.beta2))
        self.d_opt = optim.Adam(params=self.d_model.parameters(),
                                lr=float(config.d_lr), betas=(config.beta1, config.beta2))
        self.d_sche = get_lr_schedule_with_steps(config.decay_type,
                                                 self.d_opt,
                                                 drop_steps=config.drop_steps,
                                                 gamma=config.drop_gamma)
        self.g_sche = get_lr_schedule_with_steps(config.decay_type,
                                                 self.g_opt,
                                                 drop_steps=config.drop_steps,
                                                 gamma=config.drop_gamma)

        if config.float16:
            self.float16 = True
            [self.g_model, self.d_model], [self.g_opt, self.d_opt] = amp.initialize([self.g_model, self.d_model],
                                                                                    [self.g_opt, self.d_opt],
                                                                                    num_losses=2, opt_level='O1')
        else:
            self.float16 = False

    def forward(self, input, mask=None):
        if self.config.combined:
            xrec, _, _ = self.g_model(input, mask)
            m = get_combined_mask(mask)
            xrec = input * (1 - m) + xrec * m
        else:
            xrec, _ = self.g_model(input, mask)
        return xrec

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer=None):
        if last_layer is not None:
            rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            rec_grads = torch.autograd.grad(rec_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def get_losses(self, meta):
        self.iteration += 1
        real_img = meta['img']
        cb_loss = torch.tensor(0.0, device=self.config.device)
        if self.config.combined:
            fake_img, codebook_loss, combined_imgs = self.g_model(real_img, meta['mask'])
            # combined loss
            for ci in combined_imgs:
                cb_loss_ = torch.abs(F.interpolate(real_img, (ci.shape[2], ci.shape[3]), mode='area') - ci)
                mask_ = F.max_pool2d(meta['mask'], kernel_size=int(meta['mask'].shape[2] / ci.shape[2]),
                                     stride=int(meta['mask'].shape[2] / ci.shape[2]))
                # combined l1 loss
                mask_sum = torch.sum(mask_, dim=[2, 3]) + self.eps
                mask_cb_loss = torch.mean(torch.sum(cb_loss_ * mask_, dim=[2, 3]) / mask_sum)
                cb_loss += (mask_cb_loss * self.mask_cb_weight)
        else:
            fake_img, codebook_loss = self.g_model(real_img, meta['mask'])
        codebook_loss = self.codebook_weight * codebook_loss.mean()
        rec_loss = torch.mean(torch.abs(real_img - fake_img))

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(real_img, fake_img) * self.perceptual_weight
        else:
            p_loss = torch.tensor(0.0, device=self.config.device)

        disc_factor = self.adopt_weight(self.disc_factor, self.iteration,
                                        threshold=self.discriminator_iter_start)
        # discriminator loss
        d_input_real = real_img
        d_input_fake = fake_img.detach()
        d_real = self.d_model(d_input_real)
        d_fake = self.d_model(d_input_fake)
        d_real_loss = self.adversarial_loss(d_real, True, True)
        d_fake_loss = self.adversarial_loss(d_fake, False, True)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss = d_loss * disc_factor

        # generator adversarial loss
        g_input_fake = fake_img
        g_fake = self.d_model(g_input_fake)
        g_gan_loss = self.adversarial_loss(g_fake, True, False)

        if self.disc_factor < 1:
            d_weight = 1
        else:
            d_weight = self.calculate_adaptive_weight(rec_loss, g_gan_loss, last_layer=self.g_model.get_last_layer())
        g_gan_loss = g_gan_loss * disc_factor * d_weight
        g_loss = rec_loss + p_loss + g_gan_loss + codebook_loss + cb_loss

        logs = [
            ("d_loss", d_loss.item()),
            ("g_gan_loss", g_gan_loss.item()),
            ("rec_loss", rec_loss.item()),
            ("codebook_loss", codebook_loss.item()),
            ("vgg_loss", p_loss.item()),
        ]
        if self.config.combined:
            logs.append(("cb_loss", cb_loss.item()))

        return fake_img, g_loss, d_loss, logs

    def backward(self, g_loss=None, d_loss=None):
        self.d_opt.zero_grad()
        if d_loss is not None:
            if self.float16:
                with amp.scale_loss(d_loss, self.d_opt, loss_id=0) as d_loss_scaled:
                    d_loss_scaled.backward()
            else:
                d_loss.backward()
        self.d_opt.step()

        self.g_opt.zero_grad()
        if g_loss is not None:
            if self.float16:
                with amp.scale_loss(g_loss, self.g_opt, loss_id=1) as g_loss_scaled:
                    g_loss_scaled.backward()
            else:
                g_loss.backward()
        self.g_opt.step()

        self.d_sche.step()
        self.g_sche.step()

    def load(self, is_test=False, prefix=None):
        if prefix is not None:
            g_path = self.g_path + prefix + '.pth'
        else:
            g_path = self.g_path + '_last.pth'
        if self.config.restore or is_test:
            if os.path.exists(g_path):
                print('Loading %s generator...' % g_path)
                if torch.cuda.is_available():
                    data = torch.load(g_path)
                else:
                    data = torch.load(g_path, map_location=lambda storage, loc: storage)
                torch_init_model(self.g_model, g_path, 'g_model')
                if self.config.restore:
                    self.g_opt.load_state_dict(data['g_opt'])
                    # sche restore
                    from tqdm import tqdm
                    for _ in tqdm(range(data['iteration']), desc='recover sche...'):
                        self.g_sche.step()
                    self.iteration = data['iteration']
            else:
                print(g_path, 'not Found')
                raise FileNotFoundError

        d_path = self.d_path + '_last.pth'
        if self.config.restore and not is_test:
            if os.path.exists(d_path):
                print('Loading %s discriminator...' % d_path)
                if torch.cuda.is_available():
                    data = torch.load(d_path)
                else:
                    data = torch.load(d_path, map_location=lambda storage, loc: storage)
                torch_init_model(self.d_model, d_path, 'd_model')
                if self.config.restore:
                    self.d_opt.load_state_dict(data['d_opt'])
                    # sche restore
                    from tqdm import tqdm
                    for _ in tqdm(range(data['iteration']), desc='recover sche...'):
                        self.d_sche.step()
            else:
                print(d_path, 'not Found')
                raise FileNotFoundError

    def load_for_finetune(self, g_path, d_path):
        if os.path.exists(g_path):
            print('Loading %s generator...' % g_path)
            if torch.cuda.is_available():
                data = torch.load(g_path)
            else:
                data = torch.load(g_path, map_location=lambda storage, loc: storage)
            torch_init_model(self.g_model, g_path, 'g_model')
            if self.config.restore:
                self.g_opt.load_state_dict(data['g_opt'])
            self.iteration = data['iteration']
        else:
            print(g_path, 'not Found')
            raise FileNotFoundError

        if os.path.exists(d_path):
            print('Loading %s discriminator...' % d_path)
            if torch.cuda.is_available():
                data = torch.load(d_path)
            else:
                data = torch.load(d_path, map_location=lambda storage, loc: storage)
            torch_init_model(self.d_model, d_path, 'd_model')
            if self.config.restore:
                self.d_opt.load_state_dict(data['d_opt'])
        else:
            print(d_path, 'not Found')
            raise FileNotFoundError

    def save(self, prefix=None):
        if prefix is not None:
            save_g_path = self.g_path + "_{}.pth".format(prefix)
            save_d_path = self.d_path + "_{}.pth".format(prefix)
        else:
            save_g_path = self.g_path + ".pth"
            save_d_path = self.d_path + ".pth"

        print('\nsaving {} {}...\n'.format(self.name, prefix))
        torch.save({'iteration': self.iteration,
                    'g_model': self.g_model.state_dict(),
                    'g_opt': self.g_opt.state_dict()}, save_g_path)

        torch.save({'d_model': self.d_model.state_dict(),
                    'd_opt': self.d_opt.state_dict()}, save_d_path)


class VQModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ddconfig = config.model['params']['ddconfig']
        n_embed = config.model['params']['n_embed']
        embed_dim = config.model['params']['embed_dim']
        self.ddconfig['in_channels'] = 3
        self.ddconfig['out_ch'] = 3
        self.encoder = Encoder(self.ddconfig)
        self.decoder = Decoder(self.ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(self.ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.ddconfig["z_channels"], 1)

    def encode(self, x, mask=None):
        h = self.encoder(x, mask)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        if mask is not None:
            mask = F.max_pool2d(mask, kernel_size=int(mask.shape[2] / h.shape[2]),
                                stride=int(mask.shape[2] / h.shape[2]))
            quant = quant * mask + h * (1 - mask)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input, mask=None):
        quant, diff, _ = self.encode(input, mask)
        dec = self.decode(quant)
        return dec, diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class VQCombineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ddconfig = config.model['params']['ddconfig']
        n_embed = config.model['params']['n_embed']
        embed_dim = config.model['params']['embed_dim']
        self.ddconfig['in_channels'] = 3
        self.ddconfig['out_ch'] = 3
        self.encoder = Encoder(self.ddconfig)
        self.decoder = Decoder(self.ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(self.ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.ddconfig["z_channels"], 1)

    def encode(self, x, mask=None):
        h = self.encoder(x, mask)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input, mask=None):
        quant, diff, _ = self.encode(input, mask)
        dec = self.decode(quant)
        cb = input * (1 - mask) + dec * mask
        combined_imgs = []
        min_size = 16
        size = cb.shape[2]
        while size > min_size:
            size = int(size / 2)
            combined_imgs.append(F.interpolate(cb, (size, size), mode='area'))
        return dec, diff, combined_imgs

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        kw = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
