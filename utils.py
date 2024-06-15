import torch.nn.init as init
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import torch.nn.functional as F
import torch.nn as nn
import json
from torch.autograd import Variable
import torch.autograd as autograd
import os
def multinomial_loss_function(x_logit, x, z_mu, z_var, z, beta=1.):

    """
    Computes the cross entropy loss function while summing over batch dimension, not averaged!
    :param x_logit: shape: (batch_size, num_classes * num_channels, pixel_width, pixel_height), real valued logits
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param args: global parameter settings
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    batch_size = x.size(0)
    target = x
    ce = nn.MSELoss(reduction='sum')(x_logit, target)
    kl = - (0.5 * torch.sum(1 + z_var.log() - z_mu.pow(2) - z_var.log().exp()))
    loss = ce + beta * kl
    loss = loss / float(batch_size)
    ce = ce / float(batch_size)
    kl = kl / float(batch_size)

    return loss, ce, kl

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.normal_(m.weight.data,  mean=0, std=0.02)
        init.constant_(m.bias, 0.0)

def log_print(s, log):
    print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')

def synthesize_feature_test_my(netG, ae, dataset, opt,att):
    G_sample = torch.FloatTensor(opt.nSample, opt.RS_dim)
    with torch.no_grad():
        text_feat = np.tile(att.astype('float32'), (opt.nSample, 1))
        text_feat = torch.from_numpy(text_feat).to(opt.gpu)
        z = torch.randn(opt.nSample, opt.Z_dim).to(opt.gpu)
        G_sample = ae.encoder(netG.decode(z, text_feat))[:,:opt.RS_dim]
    return G_sample

def eval_relation_addloss_nell(model,ae,Ep,relationNet,FE,dataset,opt,mode ):   # initial witohut _nell
    root_path = os.path.abspath(os.path.dirname(__file__))
    model.eval()
    ae.eval()
    Ep.eval()
    if relationNet != None:
        relationNet.eval()
    print('##EVALUATING ON test DATA ——predict relation')
    hits10 = []
    hits5 = []
    hits1 = []
    mrr = []
    # 改成用关系去进行补全
    # test_candidates = json.load(open(f"../../KGC_data/{opt.dataset}/" + mode + "_candidates.json"))
    test_candidates = json.load(open(os.path.join(root_path,f"KGC_data/{opt.dataset}",mode + "_candidates.json" )))
    rela2label = {}
    rela_maxtri = np.zeros(shape=(len(test_candidates.keys()), opt.RS_dim), dtype='float32')
    label_id = 0

    for query_ in test_candidates.keys():
        description = dataset.attribute[FE.rel2id[query_]]
        rela2label[query_] = label_id
        relation_vecs = synthesize_feature_test_my(model, ae, dataset, opt, att=description)
        relation_vecs = relation_vecs.detach().cpu().numpy()
        # 对relation_vecs取平均值
        relation_vecs = relation_vecs.mean(axis=0)
        rela_maxtri[label_id] = relation_vecs
        label_id += 1

    for query_ in test_candidates.keys():
        id = rela2label[query_]
        hits10_ = []
        hits5_ = []
        hits1_ = []
        mrr_ = []
        for e1_rel, tail_candidates in test_candidates[query_].items():
            if opt.dataset == "NELL":
                head, rela, _ = e1_rel.split('\t')
            elif opt.dataset == "WiKi":
                head, rela = e1_rel.split('\t')
            
            true = tail_candidates[0]
            query_pairs = []
            query_pairs.append([FE.symbol2id[head], FE.symbol2id[true]])

            query_left = []
            query_right = []
            query_left.append(FE.ent2id[head])
            query_right.append(FE.ent2id[true])
            query = Variable(torch.LongTensor(query_pairs)).cuda()

            query_meta = FE.get_meta(query_left, query_right)
            candidate_vecs, _,_ = Ep(query, query, query_meta, query_meta)
            candidate_vecs = ae.encoder(candidate_vecs)[:, :opt.RS_dim].data.cpu().numpy()
            scores = cosine_similarity(candidate_vecs, rela_maxtri)[0]
            sort = list(np.argsort(scores))[::-1]
            rank = sort.index(id) + 1
            if rank <= 10:
                hits10.append(1.0)
                hits10_.append(1.0)
            else:
                hits10.append(0.0)
                hits10_.append(0.0)
            if rank <= 5:
                hits5.append(1.0)
                hits5_.append(1.0)
            else:
                hits5.append(0.0)
                hits5_.append(0.0)
            if rank <= 1:
                hits1.append(1.0)
                hits1_.append(1.0)
            else:
                hits1.append(0.0)
                hits1_.append(0.0)
            mrr.append(1.0 / rank)
            mrr_.append(1.0 / rank)

        # logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(query_, np.mean(hits10_), np.mean(hits5_), np.mean(hits1_), np.mean(mrr_)))
        # logging.info('Number of candidates: {}, number of text examples {}'.format(len(candidates), len(hits10_)))
        print('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(mode + query_, np.mean(hits10_),
                                                                               np.mean(hits5_), np.mean(hits1_),
                                                                               np.mean(mrr_)))

    print('############   ' + mode + '    #############')
    print('HITS10: {:.3f}'.format(np.mean(hits10)))
    print('HITS5: {:.3f}'.format(np.mean(hits5)))
    print('HITS1: {:.3f}'.format(np.mean(hits1)))
    print('MAP: {:.3f}'.format(np.mean(mrr)))
    return  np.mean(hits10),np.mean(hits5), np.mean(hits1) ,np.mean(mrr)

def eval_relation_addloss_wiki(model,ae,Ep,relationNet,FE,dataset,opt,mode):
    root_path = os.path.abspath(os.path.dirname(__file__))
    print('##EVALUATING ON test DATA ——predict relation')
    hits10 = []
    hits5 = []
    hits1 = []
    mrr = []
    # 改成用关系去进行补全
    
    test_candidates = json.load(open(os.path.join(root_path, f"KGC_data/{opt.dataset}", mode + "_tasks.json") ))
    rela2label = {}
    rela_maxtri = np.zeros(shape=(len(test_candidates.keys()), opt.RS_dim), dtype='float32')
    label_id = 0

    for query_ in test_candidates.keys():
        description = dataset.attribute[FE.rel2id[query_]]
        rela2label[query_] = label_id
        relation_vecs = synthesize_feature_test_my(model, ae, dataset, opt, att=description)
        relation_vecs = relation_vecs.detach().cpu().numpy()
        # 对relation_vecs取平均值
        relation_vecs = relation_vecs.mean(axis=0)
        rela_maxtri[label_id] = relation_vecs
        label_id += 1

    for query_ in test_candidates.keys():
        id = rela2label[query_]
        hits10_ = []
        hits5_ = []
        hits1_ = []
        mrr_ = []
        for head, rela , tail in test_candidates[query_]:
            query_pairs = []
            query_pairs.append([FE.symbol2id[head], FE.symbol2id[tail]])

            query_left = []
            query_right = []
            query_left.append(FE.ent2id[head])
            query_right.append(FE.ent2id[tail])
            query = Variable(torch.LongTensor(query_pairs)).cuda()

            query_meta = FE.get_meta(query_left, query_right)
            candidate_vecs, _,_ = Ep(query, query, query_meta, query_meta)
            candidate_vecs = ae.encoder(candidate_vecs)[:, :opt.RS_dim].data.cpu().numpy()

            scores = cosine_similarity(candidate_vecs, rela_maxtri)[0]

            sort = list(np.argsort(scores))[::-1]
            rank = sort.index(id) + 1
            if rank <= 10:
                hits10.append(1.0)
                hits10_.append(1.0)
            else:
                hits10.append(0.0)
                hits10_.append(0.0)
            if rank <= 5:
                hits5.append(1.0)
                hits5_.append(1.0)
            else:
                hits5.append(0.0)
                hits5_.append(0.0)
            if rank <= 1:
                hits1.append(1.0)
                hits1_.append(1.0)
            else:
                hits1.append(0.0)
                hits1_.append(0.0)
            mrr.append(1.0 / rank)
            mrr_.append(1.0 / rank)

        # logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(query_, np.mean(hits10_), np.mean(hits5_), np.mean(hits1_), np.mean(mrr_)))
        # logging.info('Number of candidates: {}, number of text examples {}'.format(len(candidates), len(hits10_)))
        # print('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(mode + query_, np.mean(hits10_),
        #                                                                        np.mean(hits5_), np.mean(hits1_),
        #                                                                        np.mean(mrr_)))

    print('############   ' + mode + '    #############')
    print('HITS10: {:.3f}'.format(np.mean(hits10)))
    print('HITS5: {:.3f}'.format(np.mean(hits5)))
    print('HITS1: {:.3f}'.format(np.mean(hits1)))
    print('MAP: {:.3f}'.format(np.mean(mrr)))
    return  np.mean(hits10),np.mean(hits5), np.mean(hits1) ,np.mean(mrr)

def eval_relation_addloss_GZSL(model,ae,Ep,relationNet,FE,dataset,opt,mode):
    model.eval()
    ae.eval()
    Ep.eval()
    if relationNet != None:
        relationNet.eval()
    print('##EVALUATING ON test DATA ——predict relation_GZSL')
    mrr_seen = []
    mrr_unseen = []
    # 改成用关系去进行补全

    root_path = os.path.abspath(os.path.dirname(__file__))
    # test_unseen_relation = json.load(open(f"../../KGC_data/{opt.dataset}/" + "test_tasks.json"))
    # test_seen_relation = json.load(open(f"../../KGC_data/{opt.dataset}/GZSL_data" + "/gzsl_test_data.json"))
    test_unseen_relation = json.load(open(os.path.join(root_path, f"KGC_data/{opt.dataset}/GZSL_data" + "/test_tasks.json")))
    test_seen_relation = json.load(open(os.path.join(root_path, f"KGC_data/{opt.dataset}/GZSL_data" + "/gzsl_test_data.json")))
    rela2label = {}
    rela_maxtri = np.zeros(shape=(len(test_unseen_relation.keys()) + len(test_seen_relation.keys()), opt.RS_dim), dtype='float32')
    label_id = 0

    for query_ in test_unseen_relation.keys():
        description = dataset.attribute[FE.rel2id[query_]]
        rela2label[query_] = label_id
        relation_vecs = synthesize_feature_test_my(model, ae, dataset, opt, att=description)
        relation_vecs = relation_vecs.detach().cpu().numpy()
        # 对relation_vecs取平均值
        relation_vecs = relation_vecs.mean(axis=0)
        rela_maxtri[label_id] = relation_vecs
        label_id += 1
    for query_ in test_seen_relation.keys():
        description = dataset.attribute[FE.rel2id[query_]]
        rela2label[query_] = label_id
        relation_vecs = synthesize_feature_test_my(model, ae, dataset, opt, att=description)
        relation_vecs = relation_vecs.detach().cpu().numpy()
        # 对relation_vecs取平均值
        relation_vecs = relation_vecs.mean(axis=0)
        rela_maxtri[label_id] = relation_vecs
        label_id += 1

    for query_ in test_unseen_relation.keys():
        id = rela2label[query_]
        mrr_ = []
        for head , relation , tail in test_unseen_relation[query_]:
            query_pairs = []
            query_pairs.append([FE.symbol2id[head], FE.symbol2id[tail]])
            query_left = []
            query_right = []
            query_left.append(FE.ent2id[head])
            query_right.append(FE.ent2id[tail])
            query = Variable(torch.LongTensor(query_pairs)).cuda()
            query_meta = FE.get_meta(query_left, query_right)
            candidate_vecs, _,_ = Ep(query, query, query_meta, query_meta)
            candidate_vecs = ae.encoder(candidate_vecs)[:, :opt.RS_dim].data.cpu().numpy()
            scores = cosine_similarity(candidate_vecs, rela_maxtri)[0]
            sort = list(np.argsort(scores))[::-1]
            rank = sort.index(id) + 1
            mrr_unseen.append(1.0 / rank)
            mrr_.append(1.0 / rank)

    for query_ in test_seen_relation.keys():
        id = rela2label[query_]
        mrr_ = []
        for head, relation, tail in test_seen_relation[query_]:
            query_pairs = []
            query_pairs.append([FE.symbol2id[head], FE.symbol2id[tail]])
            query_left = []
            query_right = []
            query_left.append(FE.ent2id[head])
            query_right.append(FE.ent2id[tail])
            query = Variable(torch.LongTensor(query_pairs)).cuda()
            query_meta = FE.get_meta(query_left, query_right)
            candidate_vecs, _, _ = Ep(query, query, query_meta, query_meta)
            candidate_vecs = ae.encoder(candidate_vecs)[:, :opt.RS_dim].data.cpu().numpy()
            scores = cosine_similarity(candidate_vecs, rela_maxtri)[0]
            sort = list(np.argsort(scores))[::-1]
            rank = sort.index(id) + 1
            mrr_seen.append(1.0 / rank)
            mrr_.append(1.0 / rank)
    print('MAP_unseen: {:.3f}'.format(np.mean(mrr_unseen)))
    print('MAP_seen: {:.3f}'.format(np.mean(mrr_seen)))
    S = np.mean(mrr_seen)
    U = np.mean(mrr_unseen)
    H = (2 * S * U) / (S + U)   # 谐波平均 计算均值
    print('MAP_H: {:.3f}'.format(H))
    return S, U, H