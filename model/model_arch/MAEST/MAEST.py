import torch
import torch.nn as nn
import model.model_arch.MAEST.augment as augment


class hyperTem(nn.Module):
    def __init__(self, timesteps, num_node, dim_in, dim_out, embed_dim, HT_Tem):
        super(hyperTem, self).__init__()
        self.c_out = dim_out
        self.adj = nn.Parameter(torch.randn(embed_dim, HT_Tem, timesteps), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        self.act = nn.LeakyReLU()

    def forward(self, eb, node_embeddings, time_eb):
        # H  htn   node_embeddings：Cr  adj：H——
        # n h t
        adj_dynamics = torch.einsum('nk,kht->nht', node_embeddings, self.adj).permute(1, 2, 0)
        hyperEmbeds = torch.einsum('htn,btnd->bhnd', adj_dynamics, eb)
        retEmbeds = torch.einsum('thn,bhnd->btnd', adj_dynamics.transpose(0, 1), hyperEmbeds)

        weights = torch.einsum('btd,dio->btio', time_eb, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(time_eb, self.bias_pool).unsqueeze(2)  # N, dim_out
        out = torch.einsum('btni,btio->btno', retEmbeds, weights) + bias  # b, N, dim_out
        return self.act(out + eb)


class hyperSpa(nn.Module):
    def __init__(self, num_node, dim_in, dim_out, embed_dim, HS_Spa):
        super(hyperSpa, self).__init__()
        self.c_out = dim_out
        self.adj = nn.Parameter(torch.randn(embed_dim, HS_Spa, num_node), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        self.act = nn.LeakyReLU()

    def forward(self, eb, node_embeddings, time_eb):
        adj_dynamics = torch.einsum('btk,khn->bthn', time_eb, self.adj)
        hyperEmbeds = self.act(torch.einsum('bthn,btnd->bthd', adj_dynamics, eb))
        retEmbeds = self.act(torch.einsum('btnh,bthd->btnd', adj_dynamics.transpose(-1, -2), hyperEmbeds))

        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                     #N, dim_out
        out = torch.einsum('btni,nio->btno', retEmbeds, weights) + bias     #b, N, dim_out
        return self.act(out + eb)


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


class time_feature_spg(nn.Module):
    def __init__(self, embed_dim, time_in_day=288, day_in_week=7):
        super(time_feature_spg, self).__init__()

        self.ln_day = nn.Linear(12, embed_dim)
        self.ln_week = nn.Linear(12, embed_dim)
        self.ln1 = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.Linear(embed_dim, embed_dim)
        self.act = nn.ReLU()

    def forward(self, eb):
        day = self.ln_day(eb[:, :, 0])
        week = self.ln_week(eb[:, :, 1])
        eb = self.ln(self.act(self.ln2(self.act(self.ln1(day + week)))))
        return eb


class MAEST_Encoder(nn.Module):
    def __init__(self, num_nodes, input_base_dim, input_extra_dim, hidden_dim, output_dim, horizon, embed_dim,
                 embed_dim_spa, HS, HT, HT_Tem, num_route):
        super(MAEST_Encoder, self).__init__()
        self.num_node = num_nodes
        self.input_base_dim = input_base_dim
        self.input_extra_dim = input_extra_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.embed_dim_spa = embed_dim_spa
        self.HS = HS
        self.HT = HT
        self.HT_Tem = HT_Tem
        self.num_route = num_route

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        self.node_embeddings_spg = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        self.hyperTem1 = hyperTem(horizon, num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)
        self.hyperTem2 = hyperTem(horizon, num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)
        self.hyperTem3 = hyperTem(horizon, num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)

        self.time_feature1 = time_feature(self.embed_dim)
        self.time_feature1_ = time_feature(self.embed_dim_spa)

        self.hyperSpa1 = hyperSpa(num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim_spa, self.HT_Tem)
        self.hyperSpa2 = hyperSpa(num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim_spa, self.HT_Tem)
        self.hyperSpa3 = hyperSpa(num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim_spa, self.HT_Tem)

    def forward(self, source, x_in):
        # source: B, T_1, N, D

        day_index = source[:, :, 0, self.input_base_dim: self.input_base_dim + 1]
        week_index = source[:, :, 0, self.input_base_dim + 1: self.input_base_dim + 2]

        # dt  (B, T, D)
        time_eb = self.time_feature1(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)
        teb = self.time_feature1_(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)

        x = self.hyperTem1(x_in, self.node_embeddings, time_eb)
        x = self.hyperSpa1(x, self.node_embeddings_spg, teb)

        x = self.hyperTem2(x, self.node_embeddings, time_eb)
        x = self.hyperSpa2(x, self.node_embeddings_spg, teb)

        x = self.hyperTem3(x, self.node_embeddings, time_eb)
        x = self.hyperSpa3(x, self.node_embeddings_spg, teb)

        return x


class MAEST_Decoder(nn.Module):
    def __init__(self, num_nodes, input_base_dim, input_extra_dim, hidden_dim, output_dim, horizon, embed_dim,
                 embed_dim_spa, HS, HT, HT_Tem, num_route):
        super(MAEST_Decoder, self).__init__()
        self.num_node = num_nodes
        self.input_base_dim = input_base_dim
        self.input_extra_dim = input_extra_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.embed_dim_spa = embed_dim_spa
        self.HS = HS
        self.HT = HT
        self.HT_Tem = HT_Tem
        self.num_route = num_route

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)
        self.node_embeddings_spg = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        self.hyperTem1 = hyperTem(horizon, num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim, self.HT_Tem)

        self.time_feature1 = time_feature(self.embed_dim)
        self.time_feature1_ = time_feature(self.embed_dim_spa)
        self.time_feature2 = time_feature_spg(self.embed_dim_spa)

        self.hyperSpa1 = hyperSpa(num_nodes, self.hidden_dim, self.hidden_dim, self.embed_dim_spa, self.HT_Tem)

    def forward(self, source, x_in):

        # source: B, T_1, N, D
        day_index = source[:, :, 0, self.input_base_dim:self.input_base_dim + 1]
        week_index = source[:, :, 0, self.input_base_dim + 1:self.input_base_dim + 2]

        # dt
        time_eb = self.time_feature1(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)
        teb = self.time_feature1_(torch.cat([day_index, week_index], dim=-1)).squeeze(-1)

        x = self.hyperTem1(x_in, self.node_embeddings, time_eb)
        x = self.hyperSpa1(x, self.node_embeddings_spg, teb)

        return x


class MAEST(nn.Module):
    def __init__(self, num_nodes, input_base_dim, input_extra_dim, hidden_dim, output_dim, horizon, embed_dim,
                 embed_dim_spa, HS, HT, HT_Tem, num_route
                 , mode, mask_ratio, num_remasking=1, steps_per_day=288,
                 remask_method=None, remask_ration=0.1, mask_method="flow", use_mixed_proj=True):
        super(MAEST, self).__init__()
        self.mode = mode
        self.input_base_dim = input_base_dim
        self.mask_ratio = mask_ratio
        self.use_mixed_proj = use_mixed_proj
        # self.augment = augment.gpt_agument(input_base_dim, HS, hidden_dim, embed_dim, device, num_nodes, change_epoch,
        #                            horizon, mask_ratio, ada_mask_ratio, epochs, ada_type)

        self.dim_in_flow = nn.Linear(128, hidden_dim, bias=True)
        if use_mixed_proj:
            self.dim_flow_out = nn.Linear(hidden_dim * horizon, input_base_dim * horizon, bias=True)
        else:
            self.dim_flow_out = nn.Linear(hidden_dim, input_base_dim, bias=True)

        self.mask_method = mask_method

        self.enc_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))

        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.encoder = MAEST_Encoder(num_nodes, input_base_dim, input_extra_dim, hidden_dim, output_dim, horizon,
                                  embed_dim, embed_dim_spa, HS, HT, HT_Tem, num_route)

        self.decoder = MAEST_Decoder(num_nodes, input_base_dim, input_extra_dim, hidden_dim, output_dim, horizon,
                                  embed_dim, embed_dim_spa, HS, HT, HT_Tem, num_route)
        self.init_embedding = STIF(3, 32, steps_per_day, 32,
                                   32, 0, 32,
                                   horizon, num_nodes)
        # remask
        self.remask_method = remask_method
        self.remask_ration = remask_ration
        self.num_remasking = num_remasking

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    def remask(self, flow_encode_eb, mask_idx):
        if self.remask_method == "fixed":
            flow_encode_eb = augment.fixed_remask_with_token(flow_encode_eb, self.dec_mask_token, mask_idx)
        elif self.remask_method == "random_fixed":
            flow_encode_eb = augment.fixed_random_remask_with_token(flow_encode_eb, self.dec_mask_token, mask_idx, self.remask_ration)
        elif self.remask_method == "random":
            flow_encode_eb, remask = augment.random_remask_with_token(flow_encode_eb, self.dec_mask_token, self.remask_ration)
        return flow_encode_eb

    def decoding(self, source, flow_encode_eb):
        flow_decode = self.decoder(source, flow_encode_eb)

        if self.use_mixed_proj:
            B, L, N, C = flow_decode.shape
            flow_out = self.dim_flow_out(flow_decode.permute(0, 2, 1, 3)
                                         .reshape(B, N, -1)).reshape(B, N, L, -1).permute(0, 2, 1,3)

        else:
            flow_out = self.dim_flow_out(flow_decode)

        return flow_out

    def forward_pretrain(self, source, label, batch_seen=None, epoch=None):
        # mask_source, mask, softmax_guide_weight = self.augment(source, epoch)
        if self.mask_method == "all":
            x_flow_eb = self.init_embedding(source)
            x_flow_eb = self.dim_in_flow(x_flow_eb)
            mask_source, mask, mask_idx = augment.aug_random_mask_with_token(x_flow_eb, self.enc_mask_token)
        else:
            mask_source, mask, mask_idx = augment.aug_random_mask(source, self.mask_ratio, input_dim=self.input_base_dim)
            # mask_source = mask_source[..., :self.input_base_dim]
            x_flow_eb = self.init_embedding(mask_source)
            x_flow_eb = self.dim_in_flow(x_flow_eb)

        flow_encode_eb = self.encoder(source, x_flow_eb)

        flow_encode_eb = self.encoder_to_decoder(flow_encode_eb)

        # remask
        if self.remask_method == "random":
            flow_out_list = []
            for i in range(self.num_remasking):
                flow_encode_mask = self.remask(flow_encode_eb, mask_idx)
                flow_out = self.decoding(source, flow_encode_mask)
                flow_out_list.append(flow_out)
            flow_out = flow_out_list
        else:
            flow_encode_mask = self.remask(flow_encode_eb, mask_idx)
            flow_out = self.decoding(source, flow_encode_mask)

        return flow_out, mask, 0

    def forward_fune(self, source, label):
        x_flow_eb = self.init_embedding(source)
        x_flow_eb = self.dim_in_flow(x_flow_eb)
        flow_encode_eb = self.encoder(source, x_flow_eb)
        return flow_encode_eb

    def forward(self, source, label, batch_seen=None, epoch=None):
        if self.mode == 'pretrain':
            return self.forward_pretrain(source, label, batch_seen, epoch)
        else:
            return self.forward_fune(source, label)


class STIF(nn.Module):
    def __init__(self, input_dim, input_embedding_dim, steps_per_day, tod_embedding_dim,
                 dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim,
                 in_steps, num_nodes):
        super().__init__()
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day
        self.in_steps = in_steps

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

    def forward(self, x):
        batch_size = x.shape[0]
        if self.tod_embedding_dim > 0:
            tod = x[..., -2]
        if self.dow_embedding_dim > 0:
            dow = x[..., -1]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        # (batch_size, in_steps, num_nodes, dim)
        x = torch.cat(features, dim=-1)
        return x
