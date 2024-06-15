#用预训练模型提取实体对的向量表征
import numpy as np
import torch
import json
from models import Extractor_add_loss_target
from torch.autograd import Variable
from collections import defaultdict
from tqdm import tqdm

class Feature_extracter_addloss(object):
    def __init__(self,args):
        super(Feature_extracter_addloss, self).__init__()
        self.pretrain_margin = args.pretrain_margin
        self.add_loss_weight = args.add_loss_weight
        self.loss_way = "_add_losstarget_loss" + str(args.pretrain_margin) + str(args.add_loss_weight)
        self.data_set = args.dataset
        self.data_path = args.data_path + '/'
        self.extractor_path = args.extractor_path
        self.embed_model = args.embed_model
        self.embed_dim = args.embed_dim
        self.rel2id = json.load(open(self.data_path + 'relation2ids'))
        # "concept:agriculturalproductcutintogeometricshape": 0, "concept:agentparticipatedinevent": 1
        self.ent2id = json.load(open(self.data_path + 'entity2id'))
        self.sp_dim1 = args.sp_dim1
        self.sp_dim2 = args.sp_dim2
        self.label_nums = args.y_dim
        self.latent_dim_size = args.latent_dim_size
        self.ep_dim = args.ep_dim
        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1
        self.num_ents = len(self.ent2id.keys())
        self.pad_id = self.num_symbols
        self.build_connection()
        rela_matrix = np.load(self.data_path + 'rela_matrix_.npz')['relaM']
        print('##LOADING RELATION MATRIX##')
        self.rela_matrix = rela_matrix.astype('float32')
        self.load_pretrain()

    def load_embed(self):
        symbol_id = {}
        print('##LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE']:
            embed_all = np.load(self.data_path + self.embed_model + '_embed.npz')
            ent_embed = embed_all['eM']  # entities embedding
            rel_embed = embed_all['rM']  # relation embedding

            print('    ent_embed shape is {}, the number of entity is {}'.format(ent_embed.shape, len(self.ent2id.keys())))
            print('    rel_embed shape is {}, the number of relation is {}'.format(rel_embed.shape, len(self.rel2id.keys())))

            i = 0
            embeddings = []
            for key in self.rel2id.keys():
                if key not in ['','OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[self.rel2id[key],:]))

            for key in self.ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[self.ent2id[key],:]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            # np.savez('KGC_data/NELL/Embed_used/' + self.embed_model, embeddings)
            # json.dump(symbol_id, open('KGC_data/NELL/Embed_used/' + self.embed_model +'2id', 'w'))

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings

    def load_pretrain(self):
        self.Extractor = Extractor_add_loss_target(self.embed_dim, self.num_symbols, label_nums= self.label_nums, latent_dim_size = self.latent_dim_size,
                                                   ep_dim= self.ep_dim , sp_dim1 = self.sp_dim1, sp_dim2= self.sp_dim2, embed=self.symbol2vec)
        self.Extractor.cuda()
        self.Extractor.load_state_dict(torch.load(self.extractor_path + '/intialExtractor' +self.loss_way + self.embed_model))  # 加载模型参数。
        print("pretrain_model:" + '/intialExtractor' + self.loss_way + self.embed_model)
        return
    def build_connection(self, max_=50):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.data_path + 'path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1,rel,e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                self.e1_rele2[e2].append((self.symbol2id[rel], self.symbol2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors))
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]
        return degrees
    def get_meta(self, left, right):
        left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).cuda()
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
        right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        return (left_connections, left_degrees, right_connections, right_degrees)

    def create_embed(self):

        train_tasks = json.load(open(self.data_path + 'train_tasks.json'))
        query_list = []
        query_left_list = []
        query_right_list = []
        label_list = []
        label_name_list = []
        id = 0
        for relname in train_tasks.keys():
            id +=1
            all_test_triples = train_tasks[relname]
            query_triples = all_test_triples
            for triple in query_triples:
                this_ent_pair = [self.symbol2id[triple[0]], self.symbol2id[triple[2]]]
                query_list.append(this_ent_pair)
                query_left_list.append(self.ent2id[triple[0]])
                query_right_list.append(self.ent2id[triple[2]])
                label_list.append(self.rel2id[relname])
                label_name_list.append(relname)
        unique = sorted(list(set(label_list)))
        batch_size = 256
        start = 0
        query_nums = len(query_list)
        entity_pair_Matrix = np.zeros(shape=(len(query_list), self.ep_dim), dtype='float32')
        for i in range(int(query_nums/batch_size)+1):
            if (start+batch_size) < query_nums:
                this_batch_query   = query_list[start:start+batch_size]
                this_batch_query_left  = query_left_list[start:start+batch_size]
                this_batch_query_right = query_right_list[start:start+batch_size]
            else:
                this_batch_query = query_list[start:]
                this_batch_query_left = query_left_list[start:]
                this_batch_query_right = query_right_list[start:]
            query_meta = self.get_meta(this_batch_query_left, this_batch_query_right)

            this_batch_query = Variable(torch.LongTensor(this_batch_query)).cuda()
            query_ep, _, _ = self.Extractor(this_batch_query, this_batch_query, query_meta, query_meta)
            query_ep1, _, _ = self.Extractor(this_batch_query, this_batch_query, query_meta, query_meta)
            query_ep = query_ep.cpu().detach().numpy()
            entity_pair_Matrix[start:start+batch_size] = query_ep
            start += batch_size
        np.savez("./data_presave/" + self.args.dataset + "/entity_pair_matrix_addloss", EPM=entity_pair_Matrix)
        np.savez("./data_presave/" + self.args.dataset +  "/entity_pair_label_addloss", LABEL = np.array(label_list))
        np.savez("./data_presave/" + self.args.dataset + "/entity_pair_label_name_addloss", LABEL=np.array(label_name_list))
        ################################
        print("debug")