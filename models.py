from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.z_size = args.Z_dim   # Z_dim--latent_dim
        self.args = args
        self.q_z_nn_output_dim = args.q_z_nn_output_dim  # 将编码器神经网络的输出维度设置为 args.q_z_nn_output_dim
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        # 调用 create_encoder() 创建编码器神经网络，并将返回的神经网络及其输出均值和方差分别存储在 q_z_nn、q_z_mean 和 q_z_var 中。
        self.p_x_nn, self.p_x_mean = self.create_decoder()
        # 调用 create_decoder() 创建解码器神经网络，并将返回的神经网络及其输出均值存储在 p_x_nn 和 p_x_mean 中。
        self.FloatTensor = torch.FloatTensor

    def create_encoder(self):
        # 编码器的任务是将输入数据映射到潜在空间的概率分布上，通常是高斯分布。这个高斯分布的参数包括均值和方差。
        # 通过使用神经网络来表示均值和方差，VAE 模型可以更好地学习数据的潜在结构，并生成更加准确和具有多样性的样本。
        q_z_nn = nn.Sequential(
            nn.Linear(self.args.X_dim + self.args.C_dim, self.args.vae_encoder_dim), # args.vae_encoder_dim NELL 500  WiKi 800
            nn.LeakyReLU(0.7, inplace=False),    # initial equals inplace=True
            nn.Linear(self.args.vae_encoder_dim, self.args.vae_encoder_dim), # 输入输出神经元一样多
            nn.Dropout(self.args.vae_enc_drop),
            nn.LeakyReLU(0.7, inplace=False),    # initial equals inplace=True
            nn.Linear(self.args.vae_encoder_dim, self.q_z_nn_output_dim)
        )
        q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)
        # 用于参数化一个潜在空间的高斯分布的均值和方差     通常用在变分自编码器VAE中
        q_z_var = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.z_size),
            nn.Dropout(0.2),
            nn.Softplus(),
        )
        # 在推断过程中，神经网络作为一个近似推断器，用于估计潜在空间的均值和方差。
        # 这个过程通常被称为 变分逼近，目标是找到能够最大化下界的近似后验。
        return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        p_x_nn = nn.Sequential(
            nn.Linear(self.z_size + self.args.C_dim, self.args.vae_decoder_dim), # NELL 200
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.7, inplace=True),
            nn.Linear(self.args.vae_decoder_dim, self.args.vae_decoder_dim),
            nn.BatchNorm1d(self.args.vae_decoder_dim, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.Linear(self.args.vae_decoder_dim, self.args.vae_decoder_dim),
            nn.BatchNorm1d(self.args.vae_decoder_dim, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.7, inplace=True)
        )
        p_x_mean = nn.Sequential(
            nn.Linear(self.args.vae_decoder_dim, self.args.X_dim),
            nn.LeakyReLU(0.7, inplace=True)
        )
        return p_x_nn, p_x_mean


    def encode(self, x, c):
        input = torch.cat((x,c),1)
        h = self.q_z_nn(input)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z, c):
        input = torch.cat((z, c), 1)
        h = self.p_x_nn(input)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x, c, weights=None):
        z_mu, z_var = self.encode(x, c)
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z, c)
        return x_mean, z_mu, z_var, z

class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.X_dim, args.RS_dim + args.IS_dim),
            nn.LeakyReLU(0.7, inplace=True),
            nn.Dropout(args.ae_drop)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.RS_dim + args.IS_dim, args.X_dim),
            nn.LeakyReLU(0.7, inplace=True),
            nn.Dropout(args.ae_drop),
        )
        self.f_rs = nn.Sequential(
            nn.Linear(args.RS_dim, args.RS_dim),
            nn.LeakyReLU(0.7, inplace=True),
            nn.Dropout(args.ae_drop)
        )
        self.f_is = nn.Sequential(
            nn.Linear(args.IS_dim,args.IS_dim),
            nn.LeakyReLU(0.7, inplace=True),
            nn.Dropout(args.ae_drop)
        )
    def forward(self, x):
        z = self.encoder(x)
        r_s = z[:, :self.args.RS_dim]
        i_s = z[:, self.args.RS_dim:]
        r_s_ = self.f_rs(r_s)
        i_s_ = self.f_is(i_s)
        x1 = self.decoder(torch.cat((r_s_,i_s_),1))
        return x1, z, r_s, i_s

class Extractor_add_loss_target(nn.Module):
    """
    Matching metric based on KB Embeddings
    """

    def __init__(self, embed_dim, num_symbols, label_nums, latent_dim_size, ep_dim, sp_dim1, sp_dim2, embed=None):
        super(Extractor_add_loss_target, self).__init__()
        self.embed_dim = int(embed_dim)
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(self.embed_dim, int(self.embed_dim / latent_dim_size))
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))


        self.fc1 = nn.Linear(self.embed_dim, int(self.embed_dim / latent_dim_size))
        self.fc2 = nn.Linear(self.embed_dim, int(self.embed_dim / latent_dim_size))

        self.dropout = nn.Dropout(0.2)
        self.dropout_e = nn.Dropout(0.2)

        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
        self.symbol_emb.weight.requires_grad = False
        self.support_encoder = SupportEncoder(sp_dim1, sp_dim2, dropout=0.2)
        self.logit_ = nn.Linear(ep_dim, label_nums)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        # entities=>173*50
        entities = connections[:, :, 1].squeeze(-1)
        ent_embeds = self.dropout(self.symbol_emb(entities))  # (batch, 50, embed_dim)
        concat_embeds = ent_embeds

        out = self.gcn_w(concat_embeds)
        out = torch.sum(out, dim=1)  # (batch, embed_dim)
        out = out / num_neighbors
        return out.tanh()

    def entity_encoder(self, entity1, entity2):
        entity1 = self.dropout_e(entity1)
        entity2 = self.dropout_e(entity2)
        entity1 = self.fc1(entity1)
        entity2 = self.fc2(entity2)
        entity = torch.cat((entity1, entity2), dim=-1)
        return entity.tanh()  # (batch, embed_dim)

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_e1 = self.symbol_emb(query[:, 0])  # (batch, embed_dim)
        query_e2 = self.symbol_emb(query[:, 1])  # (batch, embed_dim)
        query_e = self.entity_encoder(query_e1, query_e2)

        support_e1 = self.symbol_emb(support[:, 0])  # (batch, embed_dim)
        support_e2 = self.symbol_emb(support[:, 1])  # (batch, embed_dim)
        support_e = self.entity_encoder(support_e1, support_e2)

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_e, query_right), dim=-1)  # tanh
        support_neighbor = torch.cat((support_left, support_e, support_right), dim=-1)  # tanh

        support = support_neighbor
        query = query_neighbor

        support_g = self.support_encoder(support)  # 1 * 100
        query_g = self.support_encoder(query)

        support_g = torch.mean(support_g, dim=0, keepdim=True)

        # cosine similarity
        matching_scores = torch.matmul(query_g, support_g.t()).squeeze()

        return query_g, matching_scores , F.log_softmax(self.logit_(query_g), dim=1)
    
class SupportEncoder(nn.Module):
    """docstring for SupportEncoder"""
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(SupportEncoder, self).__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal_(self.proj1.weight)
        init.xavier_normal_(self.proj2.weight)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.proj1(x))
        output = self.dropout(self.proj2(output))
        return self.layer_norm(output + residual)
    
class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class SC_Net(nn.Module):
    def __init__(self, args):
        super(SC_Net, self).__init__()
        self.fc1 = nn.Linear(args.C_dim + args.RS_dim, args.SCN_dim)
        self.fc2 = nn.Linear(args.SCN_dim, 1)
    def forward(self, s, c, Flag):  # s:64*200 c:n_class * 300
        if Flag == "rs":
            c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)  # c_ext ： 64 * 18 * 300
            cls_num = c_ext.shape[1] #cls_num 18

            s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1) #s_ext 64 * 18 * 200
            relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1]) #
            relation = nn.ReLU()(self.fc1(relation_pairs)) # 1152 * 500
            relation = nn.Sigmoid()(self.fc2(relation))
            return relation
        elif Flag == "is":
            relation_pairs = torch.cat((s, c), dim=-1)
            relation = nn.ReLU()(self.fc1(relation_pairs))  # 1152 * 500
            relation = nn.Sigmoid()(self.fc2(relation))
            return relation

