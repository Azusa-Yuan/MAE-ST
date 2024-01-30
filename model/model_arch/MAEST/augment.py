import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F


def aug_random_mask(input_feature, drop_percent=0.2, default_value=0, input_dim=1):
    # input B L N C
    # reshape
    aug_feature = copy.deepcopy(input_feature)
    B, L, N, C = input_feature.shape
    aug_feature = aug_feature.reshape(B, -1, C)

    # generate mask_flag
    node_num = aug_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    perm = torch.randperm(node_num, device=input_feature.device)
    mask_idx = perm[: mask_num]

    # mask
    aug_feature[:, mask_idx, :input_dim] = default_value

    # generate mask_matrix
    mask = torch.ones_like(aug_feature[:, :, :input_dim])
    mask[:, mask_idx, :] = 0

    return aug_feature.reshape(B, L, N, C), 1 - mask.reshape(B, L, N, input_dim), mask_idx


def aug_random_mask_with_token(input_feature, token, drop_percent=0.2, input_dim=1):
    # input B L N C
    # reshape
    aug_feature = input_feature
    B, L, N, C = input_feature.shape
    aug_feature = aug_feature.reshape(B, -1, C)

    # generate mask_flag
    node_num = aug_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    perm = torch.randperm(node_num, device=input_feature.device)
    mask_idx = perm[: mask_num]

    # mask
    aug_feature[:, mask_idx, :] = token

    # generate mask_matrix 为了和输入的数据做对比，因此 input_dim还是需要的
    mask = torch.ones_like(aug_feature[:, :, :input_dim])
    mask[:, mask_idx, :] = 0

    # 为了方便remask 返回mask_idx
    return aug_feature.reshape(B, L, N, C), 1 - mask.reshape(B, L, N, input_dim), mask_idx


def fixed_remask_with_token(input_feature, token, mask_idx):
    # input B L N C
    # reshape
    aug_feature = input_feature
    B, L, N, C = input_feature.shape
    aug_feature = aug_feature.reshape(B, -1, C)

    # mask
    aug_feature[:, mask_idx, :] = token

    return aug_feature.reshape(B, L, N, C)


def fixed_random_remask_with_token(input_feature, token, mask_idx, drop_percent=0.2):
    # input B L N C
    # reshape
    aug_feature = input_feature
    B, L, N, C = input_feature.shape
    aug_feature = aug_feature.reshape(B, -1, C)

    # generate remask_flag
    node_num = aug_feature.shape[1]
    remask_num = int(node_num * drop_percent)

    assert remask_num <= len(mask_idx), "fixed_random_remask_with_token remask数量要少于mask在encoder"
    # 在encoder mask的列表中选取最前面mask_num个
    remask_idx = mask_idx[: remask_num]
    # mask
    aug_feature[:, remask_idx, :] = token

    return aug_feature.reshape(B, L, N, C)


def random_remask_with_token(input_feature, token, drop_percent=0.2, input_dim=1):
    # input B L N C
    # reshape
    aug_feature = input_feature.clone()
    B, L, N, C = input_feature.shape
    aug_feature = aug_feature.reshape(B, -1, C)

    # generate remask_flag
    node_num = aug_feature.shape[1]
    remask_num = int(node_num * drop_percent)
    perm = torch.randperm(node_num, device=input_feature.device)
    remask_idx = perm[: remask_num]

    # mask
    aug_feature[:, remask_idx, :] = token

    re_mask = torch.ones_like(aug_feature[:, :, :input_dim])
    re_mask[:, remask_idx, :] = 0

    return aug_feature.reshape(B, L, N, C), 1 - re_mask.reshape(B, L, N, input_dim)


class MLP_RL(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim, embed_dim, device):
        super(MLP_RL, self).__init__()

        self.ln1 = nn.Linear(dim_in, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, dim_out)

        self.weights_pool_spa = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim, hidden_dim))
        self.bias_pool_spa = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim))

        self.weights_pool_tem = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim, hidden_dim))
        self.bias_pool_tem = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim))
        self.act = nn.LeakyReLU()
        self.device = device

    # input   d_t  random_generate
    def forward(self, eb, time_eb, node_eb):
        # B T N C
        eb_out = self.ln1(eb)

        # 相当于做了一层超图卷积
        weights_spa = torch.einsum('nd,dio->nio', node_eb, self.weights_pool_spa)
        bias_spa = torch.matmul(node_eb, self.bias_pool_spa)
        out_spa = torch.einsum('btni,nio->btno', eb_out, weights_spa) + bias_spa
        # B T N C
        out_spa = self.act(out_spa)

        # 时间超图卷积
        # w_t
        weights_tem = torch.einsum('btd,dio->btio', time_eb, self.weights_pool_tem)
        # d_t
        bias_tem = torch.matmul(time_eb, self.bias_pool_tem).unsqueeze(-2)
        out_tem = torch.einsum('btni,btio->btno', out_spa, weights_tem) + bias_tem
        out_tem = self.act(out_tem)
        logits = self.ln3(out_tem)
        return logits


class time_feature(nn.Module):
    def __init__(self, embed_dim):
        super(time_feature, self).__init__()

        # z_t
        self.ln_day = nn.Linear(1, embed_dim)
        self.ln_week = nn.Linear(1, embed_dim)
        self.ln1 = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, eb):
        # 代码堪忧
        day = self.ln_day(eb[:, :, 0:1])
        week = self.ln_week(eb[:, :, 1:2])
        # dt
        eb = self.ln(self.act(self.ln2(self.act(self.ln1(day + week)))))
        return eb


class gpt_agument(nn.Module):
    def __init__(self, input_base_dim, HS, hidden_dim, embed_dim, device, num_nodes, change_epoch,
                 horizon, mask_ratio, ada_mask_ratio, epochs, ada_type, scaler_zeros=0):
        super().__init__()
        self.HS = HS
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.device = device
        self.num_node = num_nodes
        self.change_epoch = change_epoch
        self.horizon = horizon
        self.mask_ratio = mask_ratio
        self.input_base_dim = input_base_dim
        self.ada_mask_ratio = ada_mask_ratio
        self.epochs = epochs
        self.scaler_zeros = scaler_zeros
        self.ada_type = ada_type

        self.MLP_RL = MLP_RL(input_base_dim, self.HS, self.hidden_dim, self.embed_dim, self.device)
        # temporal embedding for mask
        self.teb4mask = time_feature(self.embed_dim)
        # noding embedding for mask
        self.neb4mask = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

    def forward(self, source, epoch=None):
        if epoch <= self.change_epoch:
            # random mask   reshape(-1) 变成一个维度
            mask_random_init = torch.rand_like(source[..., 0:self.input_base_dim].reshape(-1)).to('cuda:0')
            # acquire mask id
            _, max_idx_random = torch.sort(mask_random_init, dim=0, descending=True)
            mask_num = int(mask_random_init.shape[0] * self.mask_ratio)
            max_idx = max_idx_random[:mask_num]  # NYC_TAXI
            mask_random = torch.ones_like(max_idx_random)
            mask_random = mask_random.scatter_(0, max_idx, 0)
            mask_random = mask_random.reshape(-1, self.horizon, self.num_node, self.input_base_dim)
            final_mask = mask_random

            #  B T 1 1
            day_index_ori = source[:, :, 0, self.input_base_dim:self.input_base_dim + 1]
            week_index_ori = source[:, :, 0, self.input_base_dim + 1:self.input_base_dim + 2]
            # get d_t  B T 1 embed_dim
            time_eb_logits = self.teb4mask(torch.cat([day_index_ori, week_index_ori], dim=-1))
            # 类似聚合操作
            guide_weight = self.MLP_RL(source[..., 0:self.input_base_dim], time_eb_logits, self.neb4mask)

            # get the classification label

            softmax_guide_weight = F.softmax(guide_weight, dim=-1)
        else:
            ### intra-class  inter-class

            # get the HS first
            day_index_ori = source[:, :, 0, self.input_base_dim:self.input_base_dim + 1]
            week_index_ori = source[:, :, 0, self.input_base_dim + 1:self.input_base_dim + 2]
            time_eb_logits = self.teb4mask(torch.cat([day_index_ori, week_index_ori], dim=-1))
            guide_weight = self.MLP_RL(source[..., 0:self.input_base_dim], time_eb_logits, self.neb4mask)

            # get the classification label
            # B T N H_S   损失函数用模型产生的聚类和这里引导权重的聚类做一个分布上的kl
            softmax_guide_weight = F.softmax(guide_weight, dim=-1)
            # 产生排序后的值和索引
            max_value, max_idx_all = torch.sort(softmax_guide_weight, dim=-1, descending=True)
            # 选取最大的作为他们的聚合分类
            label_c = max_idx_all[..., 0].reshape(-1)  # [batch_size, time_steps, num_node]

            # calculate number of random mask and adaptive mask
            train_process = ((epoch - self.change_epoch) / (self.epochs - self.change_epoch)) * self.ada_mask_ratio
            if train_process > 1:
                train_process = 1
            mask_num_sum = int(source[:, :, :, 0].reshape(-1).shape[0] * self.mask_ratio)
            adaptive_mask_num = int(mask_num_sum * train_process)
            random_mask_num = mask_num_sum - adaptive_mask_num

            ### adaptive mask  随机选取聚类的种类
            # random choose mask class until the adaptive_mask_num<=select_num
            list_c = list(range(0, self.HS))
            random.shuffle(list_c)
            select_c = torch.zeros_like(label_c).to(self.device)
            select_d = torch.zeros_like(label_c).to(self.device)
            select_f = torch.zeros_like(label_c).to(self.device)

            # i指选择的聚类种类，select_num 为选择mask的地方
            select_num = 0
            i = 0
            if self.ada_type == 'all':
                while select_num < adaptive_mask_num:
                    select_c[label_c == list_c[i]] = 1
                    select_num = torch.sum(select_c)
                    i = i + 1
                # 需要掩盖两种聚类以上时，就需要使用不同种类的mask
                if i >= 2:
                    for k in range(i - 1):
                        select_d[label_c == list_c[k]] = 1
                    adaptive_dnum = torch.sum(select_d)
                    select_f[label_c == list_c[i - 1]] = 1
                else:
                    adaptive_dnum = 0
                    select_f = select_c.clone()
            else:
                while select_num < adaptive_mask_num:
                    select_c[label_c == list_c[i]] = 1
                    select_num = torch.sum(select_c)
                    i = i + 1
                adaptive_dnum = 0
                select_f = select_c.clone()

            # randomly choose top adaptive_mask_num to mask
            select_f = select_f.reshape(-1, self.horizon * self.num_node).reshape(-1)
            select_d = select_d.reshape(-1, self.horizon * self.num_node).reshape(-1)
            mask_adaptive_init = torch.rand_like(source[..., 0:1].reshape(-1)).to('cuda:0')
            # 相当于在f类随机挑选
            mask_adaptive_init = select_f * mask_adaptive_init
            _, max_idx_adaptive = torch.sort(mask_adaptive_init, dim=0, descending=True)

            select_idx_adaptive = max_idx_adaptive[:(adaptive_mask_num - adaptive_dnum)]

            mask_adaptive = torch.ones_like(max_idx_adaptive)
            mask_adaptive = mask_adaptive.scatter_(0, select_idx_adaptive, 0)

            # 屏蔽掉所有分配到D的
            mask_adaptive = mask_adaptive * (1 - select_d)

            # random mask
            mask_random_init = torch.rand_like(source[..., 0:1].reshape(-1)).to('cuda:0')
            mask_random_init = mask_adaptive * mask_random_init
            _, max_idx_random = torch.sort(mask_random_init, dim=0, descending=True)

            select_idx_random = max_idx_random[:random_mask_num]
            mask_random = torch.ones_like(max_idx_random)
            mask_random = mask_random.scatter_(0, select_idx_random, 0)
            mask_random = mask_random.reshape(-1, self.horizon * self.num_node).reshape(-1, self.horizon,
                                                                                        self.num_node)

            # final_mask
            mask_adaptive = mask_adaptive.reshape(-1, self.horizon * self.num_node).reshape(-1, self.horizon,
                                                                                            self.num_node)
            final_mask = (mask_adaptive * mask_random).unsqueeze(-1)
            if self.input_base_dim != 1:
                final_mask = final_mask.repeat(1, 1, 1, self.input_base_dim)

        final_mask = final_mask.detach()
        mask_source = final_mask * source[..., 0:self.input_base_dim]
        mask_source[final_mask == 0] = self.scaler_zeros
        # 输入转化为和hidden dim相同的维度

        return mask_source, final_mask, softmax_guide_weight


def test_aug_random_mask():
    x = torch.randn(4, 4, 4, 2)
    x = aug_random_mask(x)
    print(x)

# class gptst_aug(nn.Module):
#     super().__init__()

# def gptst_aug():


# test_aug_random_mask()
