import torch
import torch.nn as nn
import os
from src.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
from src.vqgan_models import VQModel, VQCombineModel
from src.skvqgan_models import SKVQModel
from src.transformer import GPT
from utils.utils import torch_show_all_params, torch_init_model
from utils.utils import Config
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np

try:
    from apex import amp
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class GETransformer(nn.Module):
    def __init__(self, config, sketch_path, g_path, logger=None):
        super(GETransformer, self).__init__()
        self.config = config
        self.iteration = 0
        self.name = config.model_type
        self.sketch_config = Config(os.path.join(sketch_path, 'sketch_config.yml'))
        self.g_config = Config(os.path.join(g_path, 'config.yml'))
        self.sketch_path = os.path.join(sketch_path, self.sketch_config.model_type + '_g')
        self.g_path = os.path.join(g_path, self.g_config.model_type + '_g')
        self.transformer_path = os.path.join(config.path, self.name)
        self.trans_size = config.trans_size
        self.eps = 1e-6

        self.sketch_model = SKVQModel(self.sketch_config).to(config.device)
        if config.combined:
            self.g_model = VQCombineModel(self.g_config).to(config.device)
        else:
            self.g_model = VQModel(self.g_config).to(config.device)
        self.sketch_model.eval()
        self.g_model.eval()
        self.transformer = GPT(config).to(config.device)
        self.ignore_idx = -1
        self.nll_loss = CrossEntropyLoss(ignore_index=self.ignore_idx)

        if config.init_gpt_with_vqvae:
            self.transformer.z_emb.weight = self.g_model.quantize.embedding.weight
            self.transformer.c_emb.weight = self.sketch_model.quantize.embedding.weight

        if logger is not None:
            logger.info('Sketch Parameters:{}'.format(torch_show_all_params(self.sketch_model)))
            logger.info('Gen Parameters:{}'.format(torch_show_all_params(self.g_model)))
            logger.info('Transformer Parameters:{}'.format(torch_show_all_params(self.transformer)))
        else:
            print('Sketch Parameters:{}'.format(torch_show_all_params(self.sketch_model)))
            print('Gen Parameters:{}'.format(torch_show_all_params(self.g_model)))
            print('Transformer Parameters:{}'.format(torch_show_all_params(self.transformer)))

        # loss
        no_decay = ['bias', 'ln1.bias', 'ln1.weight', 'ln2.bias', 'ln2.weight']
        param_optimizer = self.transformer.named_parameters()
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any([nd in n for nd in no_decay])],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any([nd in n for nd in no_decay])],
             'weight_decay': 0.0}
        ]
        self.opt = AdamW(params=optimizer_parameters,
                         lr=float(config.lr), betas=(config.beta1, config.beta2))
        self.sche = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_iters,
                                                    num_training_steps=config.max_iters)

        if config.float16:
            self.float16 = True
            [self.sketch_model, self.g_model, self.transformer], self.opt \
                = amp.initialize([self.sketch_model, self.g_model, self.transformer], self.opt, opt_level='O1')
        else:
            self.float16 = False

    @torch.no_grad()
    def encode_to_z(self, x, mask=None):
        quant_z, _, info = self.g_model.encode(x, mask)  # [B,D,H,W]
        indices = info[2].view(quant_z.shape[0], -1)  # [B,L]
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        quant_c, _, info, _ = self.sketch_model.encode(c)
        indices = info[2].view(quant_c.shape[0], -1)  # [B,L]
        return quant_c, indices

    def build_mask(self, mask, return_simple=False):
        '''
        mask:
        [0,1,1]
        [0,1,1]
        [0,0,0]
        ->reshape
        [0,1,1,0,1,1,0,0,0]
        ->1-
        [1,0,0,1,0,0,1,1,1]
        ->mask_t
        [1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1]
        ->+tril(9,9)*mask_t
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1, 1]]
        ->+mask_t+clip[0~1]
        [1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1]
        total_mask:
        [C2C,C2Z]
        [Z2C,Z2C]
        ->
        [1,0]    [M,0]
        [1,M] or [1,M] ?
        '''
        B = mask.shape[0]
        L = self.trans_size * self.trans_size
        mask = 1 - mask.reshape(B, 1, L)  # [B,1,L]
        # pad mask 由于z和c均有一个pad_idx，所以pad一个mask=1
        pad_mask = torch.ones((B, 1, 1)).to(dtype=mask.dtype, device=mask.device)
        mask = torch.cat([pad_mask, mask], dim=2)  # [B,1,L+1]
        mask_t = mask.transpose(1, 2).repeat(1, 1, L + 1)  # [B,L+1,1]->[B,L+1,L+1]
        mask = mask.repeat(1, L + 1, 1)  # [B,L+1,L+1]
        mask_t = mask_t * mask + (1 - mask_t)
        tril_mask = torch.tril(torch.ones_like(mask))
        tril_plm_mask = tril_mask * mask_t
        M = torch.clamp(mask + tril_plm_mask, 0, 1)
        if return_simple:
            return M, tril_mask
        else:
            plm_mask_p1 = torch.cat([torch.ones_like(M), torch.zeros_like(M)], dim=2)  # [B,L+1,2L+2]
            plm_mask_p2 = torch.cat([torch.ones_like(M), M], dim=2)  # [b,L+1,2L+2]
            plm_mask = torch.cat([plm_mask_p1, plm_mask_p2], dim=1)  # [B,2L+2,2L+2]

            lm_mask_p1 = torch.cat([torch.ones_like(M), torch.zeros_like(M)], dim=2)
            lm_mask_p2 = torch.cat([torch.ones_like(M), tril_mask], dim=2)
            lm_mask = torch.cat([lm_mask_p1, lm_mask_p2], dim=1)  # [B,2L+2,2L+2]
            return plm_mask, lm_mask

    def forward(self, z_indices, c_indices, mask=None, mc=None):
        # z_indices:[B,L]
        # c_indices:[B,L]
        # mask:[B,1,L',L'] (L=L'*L')
        # build transformer mask
        plm_mask, lm_mask = self.build_mask(mask, return_simple=False)  # [B,2L+2,2L+2]
        if mc is None:
            trans_mask = plm_mask
        else:
            mc = mc.reshape(-1, 1, 1)
            trans_mask = lm_mask * mc + plm_mask * (1 - mc)

        logits_c, logits_z = self.transformer(c_indices, z_indices, mask=trans_mask)
        return logits_c, logits_z

    def perplexity(self, x, c, mask):
        _, z_indices = self.encode_to_z(x, mask)
        _, c_indices = self.encode_to_c(c)
        mask = F.max_pool2d(mask, kernel_size=int(mask.shape[2] / self.trans_size),
                            stride=int(mask.shape[2] / self.trans_size))
        plm_mask, _ = self.build_mask(mask, return_simple=False)  # [B,2L+2,2L+2]
        _, logits_z = self.transformer(c_indices, z_indices, mask=plm_mask)
        plm_pos = mask.reshape(mask.shape[0], self.trans_size * self.trans_size).cpu().numpy()
        # get log softmax [B,L+1,V]->[B,L,V]
        logits_z = logits_z[:, :-1, :]
        [B, L, _] = logits_z.shape
        log_probs_z = F.log_softmax(logits_z, dim=2).cpu().numpy()
        z_tar = z_indices.cpu().numpy()
        nlls = []
        for i in range(B):
            for j in range(L):
                if plm_pos[i, j] == 1:  # 只计算被mask区域的困惑度
                    nlls.append(-1 * log_probs_z[i, j, z_tar[i, j]])
        ppl = np.exp(np.mean(nlls))

        return ppl

    def get_losses(self, meta):
        self.iteration += 1
        mask = meta['mask']
        quant_z, z_indices = self.encode_to_z(meta['img'], mask)
        quant_c, c_indices = self.encode_to_c(meta['sketch'])
        # mc:[B,]1表示lm，0表示plm
        if self.config.resize_type is None:
            mask = F.max_pool2d(mask, kernel_size=int(mask.shape[2] / self.trans_size),
                                stride=int(mask.shape[2] / self.trans_size))
        else:
            mask = F.interpolate(mask, (16, 16), mode=self.config.resize_type)
        logits_c, logits_z = self.forward(z_indices, c_indices, mask, meta['mc'])
        loss = 0
        nll_loss = 0
        # 只计算被mask区域的loss,[B,L]
        plm_pos = mask.reshape(mask.shape[0], self.trans_size * self.trans_size)
        if meta['mc'] is not None:
            mc = meta['mc'].reshape(-1, 1)
            loss_pos = torch.ones_like(plm_pos) * mc + plm_pos * (1 - mc)
        else:
            loss_pos = plm_pos
        if self.config.plm_in_cond:
            c_pred = logits_c[:, :-1, :]  # 左移一位
            # 吧不需要优化loss的tar位置变为-1
            c_tar = torch.clamp_min(c_indices - 5000 * (1 - plm_pos),
                                    self.ignore_idx).to(dtype=torch.long)
            c_loss = self.nll_loss(c_pred.reshape(-1, c_pred.shape[-1]), c_tar.reshape(-1))
            nll_loss += c_loss
        else:
            c_loss = None
        z_pred = logits_z[:, :-1, :]  # [B,L+1,V]
        z_tar = torch.clamp_min(z_indices - 5000 * (1 - loss_pos),
                                self.ignore_idx).to(dtype=torch.long)
        z_loss = self.nll_loss(z_pred.reshape(-1, z_pred.shape[-1]), z_tar.reshape(-1))
        nll_loss += z_loss
        loss += nll_loss

        logs = [("z_loss", z_loss.item())]
        if c_loss is not None:
            logs.append(("c_loss", c_loss.item()))

        return loss, logs

    def backward(self, loss=None):
        self.opt.zero_grad()
        if loss is not None:
            if self.float16:
                with amp.scale_loss(loss, self.opt) as loss_scaled:
                    loss_scaled.backward()
            else:
                loss.backward()
        self.opt.step()
        self.sche.step()

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    # 暂不支持batch编辑
    @torch.no_grad()
    def sample(self, x, c, mask, temperature=1.0, greed=True, top_k=None):
        '''
        :param x:[1,3,H,W] image
        :param c:[1,X,H,W] condition
        :param mask: [1,1,H,W] mask
        '''
        if mask is None:
            mask = torch.ones((1, 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        mask_origin = mask.clone()
        quant_z, z_indices = self.encode_to_z(x, mask)
        quant_c, c_indices = self.encode_to_c(c)

        if self.config.resize_type is None:
            mask = F.max_pool2d(mask, kernel_size=int(mask.shape[2] / self.trans_size),
                                stride=int(mask.shape[2] / self.trans_size))
        else:
            mask = F.interpolate(mask, (16, 16), mode=self.config.resize_type)
        output_pos = torch.where(mask.reshape(-1) == 1)[0].cpu().tolist()  # [HW,]
        steps = int(torch.sum(mask).cpu().numpy())  # step等于被mask的patch数量
        assert len(output_pos) == steps
        mask_ = mask.clone()

        # 自回归生成编辑区域
        for i in range(steps):
            plm_mask, _ = self.build_mask(mask_, return_simple=False)  # [1,2L+2,2L+2]
            # logits是左移后的[1,L+1]
            _, logits_z = self.transformer(c_indices, z_indices, mask=plm_mask)
            # pluck the logits at the output position i
            logits = logits_z[:, output_pos[i], :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)  # [1,V]
            # sample from the distribution or take the most likely
            if not greed:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # set the value to the sequence and continue
            z_indices[0, output_pos[i]] = ix
            # refine the mask to 0 (has been generated)
            mask_[0, 0, output_pos[i] // self.trans_size, output_pos[i] % self.trans_size] = 0
        # 最终确保所有mask_均为0，即编辑区域全部生成完毕
        assert torch.sum(mask_) == 0

        # 重新使用z_indices得到quant_z2 [1,D,H,W]
        bhwc = (quant_z.shape[0], quant_z.shape[2], quant_z.shape[3], quant_z.shape[1])
        quant_z2 = self.g_model.quantize.get_codebook_entry(z_indices.reshape(-1),
                                                            shape=bhwc)
        # 根据mask合并
        if self.config.combined_vqgan:
            from utils.utils import get_combined_mask
            c = get_combined_mask(mask_origin)
            xrec = self.g_model.decode(quant_z2)
            output = x * (1 - c) + xrec * c
        else:
            quant_z = quant_z * (1 - mask) + quant_z2 * mask
            output = self.g_model.decode(quant_z)

        return output

    def load(self, is_test=False, prefix=None):
        if prefix is not None:
            transformer_path = self.transformer_path + prefix + '.pth'
        else:
            transformer_path = self.transformer_path + '_last.pth'
        if self.config.restore or is_test:
            if os.path.exists(transformer_path):
                print('Loading %s Transformer...' % transformer_path)
                if torch.cuda.is_available():
                    data = torch.load(transformer_path)
                else:
                    data = torch.load(transformer_path, map_location=lambda storage, loc: storage)
                torch_init_model(self.transformer, transformer_path, 'model')

                if self.config.restore:
                    self.opt.load_state_dict(data['opt'])
                    # sche restore
                    from tqdm import tqdm
                    for _ in tqdm(range(data['iteration']), desc='recover sche...'):
                        self.sche.step()
                    self.iteration = data['iteration']
            else:
                print(transformer_path, 'not Found')
                raise FileNotFoundError

    def restore_from_stage1(self, prefix=None):
        sketch_path = self.sketch_path + '_last.pth'
        if os.path.exists(sketch_path):
            print('Loading %s Sketch...' % sketch_path)
            torch_init_model(self.sketch_model, sketch_path, 'g_model')
        else:
            print(sketch_path, 'not Found')
            raise FileNotFoundError

        if prefix is not None:
            g_path = self.g_path + prefix + '.pth'
        else:
            g_path = self.g_path + '_last.pth'
        if os.path.exists(g_path):
            print('Loading %s G...' % g_path)
            torch_init_model(self.g_model, g_path, 'g_model')
        else:
            print(g_path, 'not Found')
            raise FileNotFoundError

    def save(self, prefix=None):
        if prefix is not None:
            save_path = self.transformer_path + "_{}.pth".format(prefix)
        else:
            save_path = self.transformer_path + ".pth"

        print('\nsaving {} {}...\n'.format(self.name, prefix))
        torch.save({'iteration': self.iteration,
                    'model': self.transformer.state_dict(),
                    'opt': self.opt.state_dict()}, save_path)
