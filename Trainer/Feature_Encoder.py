# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from args_NELL import read_options
from args_WiKi import read_options_WiKi
from data_loader import *
from models import Extractor_add_loss_target
from utils import weights_init
from Texting_Embedding_NELL import NELL_text_embedding
from Texting_Embedding_WiKi import WiKi_text_embedding
import random
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))



class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        for k, v in vars(args).items(): setattr(self, k, v)
        self.args = args
        ##########
        self.pretrain_feature_extractor = True
        self.loss_way = "_add_losstarget_loss" + str(self.pretrain_margin) + str(self.add_loss_weight)
        ##########
        self.data_path = os.path.join(root_path, "KGC_data", self.dataset)

        self.train_tasks = json.load(open(os.path.join(self.data_path,'train_tasks.json')))
        self.rel2id = json.load(open(os.path.join(self.data_path,'relation2ids')))

        # Generate the relation matrix according to word embeddings and TFIDF
        self.generate_text_embedding = True
        if self.generate_text_embedding:
            if self.dataset == "NELL":
                print("NELL_text_embedding is running!")
                NELL_text_embedding(data_path=self.data_path)
            elif self.dataset == "WiKi":
                print("WiKi_text_embedding is running!")
                WiKi_text_embedding(data_path=self.data_path)
            else:
                raise AttributeError("wrong dataset name!")   #  initial without WiKi dataset
        rela_matrix = np.load(os.path.join(self.data_path , 'rela_matrix_without_description.npz'))['relaM']  # initials equal 'rela_matrix.npz'
        print('##LOADING RELATION MATRIX##')
        self.rela_matrix = rela_matrix.astype('float32')
        self.ent2id = json.load(open(os.path.join(self.data_path , 'entity2id')))

        print('##LOADING CANDIDATES ENTITIES##')
        self.rel2candidates = json.load(open(os.path.join(self.data_path, 'rel2candidates_all.json')))
        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(os.path.join(self.data_path,'e1rel_e2_all.json')))

        noises = Variable(torch.randn(self.test_sample, self.noise_dim)).cuda()
        self.test_noises = 0.1 * noises

        self.meta = not self.no_meta
        self.label_num = len(self.train_tasks.keys())

        self.rela2label = dict()
        rela_sorted = sorted(list(self.train_tasks.keys()))
        for i, rela in enumerate(rela_sorted):
            self.rela2label[rela] = int(i)

        print('##LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1  #
        self.pad_id = self.num_symbols

        print('##DEFINE FEATURE EXTRACTOR')
        self.Extractor = Extractor_add_loss_target(self.embed_dim, self.num_symbols, label_nums= self.label_num, latent_dim_size = self.latent_dim_size, ep_dim= self.ep_dim , sp_dim1 = self.sp_dim1, sp_dim2= self.sp_dim2, embed=self.symbol2vec)
        self.Extractor.cuda()
        self.Extractor.apply(weights_init)
        self.E_parameters = filter(lambda p: p.requires_grad, self.Extractor.parameters())
        self.optim_E = optim.Adam(self.E_parameters, lr=self.lr_E)

        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim_E, milestones=[50000], gamma=0.5)

        print('##DEFINE GENERATOR')
        self.num_ents = len(self.ent2id.keys())

        print('##BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)
    def load_symbol2id(self):
        symbol_id = {}
        i = 0
        for key in self.rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in self.ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def load_embed(self):

        symbol_id = {}

        print('##LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            embed_all = np.load(os.path.join(self.data_path , self.embed_model +'_embed.npz'))
            ent_embed = embed_all['eM']
            rel_embed = embed_all['rM']

            if self.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            print('    ent_embed shape is {}, the number of entity is {}'.format(ent_embed.shape,
                                                                                 len(self.ent2id.keys())))
            print('    rel_embed shape is {}, the number of relation is {}'.format(rel_embed.shape,
                                                                                   len(self.rel2id.keys())))

            i = 0
            embeddings = []
            for key in self.rel2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[self.rel2id[key], :]))

            for key in self.ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[self.ent2id[key], :]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            self.symbol2id = symbol_id
            self.symbol2vec = embeddings

    #  build neighbor connection
    def build_connection(self, max_=100):

        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(os.path.join(self.data_path,"path_graph")) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                # self.e1_rele2[e2].append((self.symbol2id[rel+'_inv'], self.symbol2id[e1]))
                self.e1_rele2[e2].append((self.symbol2id[rel], self.symbol2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors))
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]
        # json.dump(degrees, open(self.dataset + '/degrees', 'w'))
        # assert 1==2

        return degrees

    def save_pretrain(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.Extractor.state_dict(), path + 'Extractor_add_loss' + self.loss_way + str(self.embed_model))

    def get_meta(self, left, right):
        left_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).cuda()
        test_1 = np.stack([self.connections[_, :, :] for _ in left], axis=0)
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
        right_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        return (left_connections, left_degrees, right_connections, right_degrees)

    def get_meta_cpu(self, left, right):
        left_connections = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0)))
        test_1 = np.stack([self.connections[_, :, :] for _ in left], axis=0)
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left]))
        right_connections = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0)))
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right]))
        return (left_connections, left_degrees, right_connections, right_degrees)

    def pretrain_Extractor(self):
        print('\n##PRETRAINING FEATURE EXTRACTOR ....')
        pretrain_losses = deque([], 100)
        class_losses = deque([], 100)
        marge_losses = deque([], 100)
        i = 0

        for data in self.Extractor_generate_add_label(self.data_path, self.pretrain_batch_size, self.symbol2id, self.ent2id,
                                                 self.e1rel_e2, self.pretrain_few, self.pretrain_subepoch,
                                                 self.rela2label):
            i += 1

            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right, label_list = data

            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).cuda()
            query = Variable(torch.LongTensor(query)).cuda()
            false = Variable(torch.LongTensor(false)).cuda()
            label_list = Variable(torch.LongTensor(label_list)).cuda()
            query_ep, query_scores, predict = self.Extractor(query, support, query_meta, support_meta)
            false_ep, false_scores, predict = self.Extractor(false, support, false_meta, support_meta)

            margin_ = query_scores - false_scores
            pretrain_loss = F.relu(self.pretrain_margin - margin_).mean()
            pretrain_loss += self.add_loss_weight * F.nll_loss(predict, label_list)
            self.optim_E.zero_grad()
            pretrain_loss.backward()
            pretrain_losses.append(pretrain_loss.item())

            if i % self.pretrain_loss_every == 0:
                # print("Step: %d, Feature Extractor Pretraining loss: %.2f ,marge_loss %.2f ,class_loss %.2f" % (i, np.mean(pretrain_losses,np.mean(marge_losses),np.mean(class_losses))))
                print("Step: %d, Feature Extractor Pretraining loss: %.2f" % (i, np.mean(pretrain_losses)))

            self.optim_E.step()

            if i > self.pretrain_times:
                break

        # self.save_pretrain()
        print('SAVE FEATURE EXTRACTOR PRETRAINING MODEL!!!')

    def Extractor_generate_add_label(self,dataset, batch_size, symbol2id, ent2id, e1rel_e2, few, sub_epoch, rela2label):

        print('\nLOADING PRETRAIN TRAINING DATA')
        train_tasks = json.load(open(dataset + '/train_tasks.json'))
        rel2candidates = json.load(open(dataset + '/rel2candidates_all.json'))
        task_pool = train_tasks.keys()
        t_num = list()
        for k in task_pool:
            if len(rel2candidates[k]) <= 20:
                v = 0
            else:
                v = min(len(rel2candidates[k]), 1000)
            t_num.append(v)
        t_sum = sum(t_num)
        probability = [float(item) / t_sum for item in t_num]
        while True:
            support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right, label_list = \
                list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
            for _ in range(sub_epoch):

                query = random_pick(task_pool, probability)

                label_list += [rela2label[query] for i in range(batch_size)]

                candidates = rel2candidates[query]

                train_and_test = train_tasks[query]

                random.shuffle(train_and_test)

                support_triples = train_and_test[:few]

                support_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

                support_left += [ent2id[triple[0]] for triple in support_triples]
                support_right += [ent2id[triple[2]] for triple in support_triples]

                all_test_triples = train_and_test[few:]

                if len(all_test_triples) == 0:
                    continue

                if len(all_test_triples) < batch_size:
                    query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
                else:
                    query_triples = random.sample(all_test_triples, batch_size)

                query_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

                query_left += [ent2id[triple[0]] for triple in query_triples]
                query_right += [ent2id[triple[2]] for triple in query_triples]

                for triple in query_triples:
                    e_h = triple[0]
                    rel = triple[1]
                    e_t = triple[2]
                    while True:
                        noise = random.choice(candidates)
                        if noise in ent2id.keys():  # ent2id.has_key(noise):
                            if (noise not in e1rel_e2[e_h + rel]) and noise != e_t:
                                break
                    false_pairs.append([symbol2id[e_h], symbol2id[noise]])
                    false_left.append(ent2id[e_h])
                    false_right.append(ent2id[noise])

            yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right, label_list
    def load_pretrain(self):
        # self.Extractor = Extractor_add_loss_target(self.embed_dim, self.num_symbols, label_nums= self.label_nums, latent_dim_size = self.latent_dim_size, ep_dim= self.ep_dim , sp_dim1 = self.sp_dim1, sp_dim2= self.sp_dim2, embed=self.symbol2vec)
        # self.Extractor.cuda()
        self.Extractor.load_state_dict(torch.load(os.path.join(root_path, "saved_model",self.dataset,'intialExtractor' +self.loss_way + self.embed_model)) )
        print("pretrain_model:" + '/intialExtractor' + self.loss_way + self.embed_model)
        return
    
    def train(self):
        print('\n##START ADVERSARIAL TRAINING...')

        # Pretraining step to obtain reasonable real data embeddings
        self.pretrain_feature_extractor = False
        if self.pretrain_feature_extractor:
            self.pretrain_Extractor()
            print('Finish Pretraining!\n')
        self.load_pretrain()
if __name__ == '__main__':

    args = read_options_WiKi()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.device)
        print("GPU is availabqle!")
    else:
        print("GPU is not available!")

    # setup random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = Trainer(args)
    # trainer.train()  initial without #