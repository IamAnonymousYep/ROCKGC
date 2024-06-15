import json
import random
import numpy as np
import os
def train_generate_decription(dataset, batch_size, symbol2id, ent2id, e1rel_e2, rel2id, args, rela2label, rela_matrix):
    print('##LOADING TRAINING DATA')
    train_tasks = json.load(open(os.path.join(dataset,'train_tasks.json')))

    # train_tasks = json.load(open(dataset + 'train_tasks.json'))
    print('##LOADING CANDIDATES')
    rel2candidates = json.load(open(os.path.join(dataset , 'rel2candidates_all.json')))

    task_pool = list(train_tasks.keys())
    while True:
        count_sample_all = 0
        rel_batch, query_pairs, query_left, query_right, false_pairs, false_left, false_right, labels = [], [], [], [], [], [], [], []

        random.shuffle(task_pool)
        for query in task_pool:
            this_query_sample = random.choice([2,3,4,5])
            if count_sample_all + this_query_sample > batch_size:
                this_query_sample = batch_size - count_sample_all
            count_sample_all += this_query_sample
            candidates = rel2candidates[query]
            if len(candidates) <= 20:
                # print 'not enough candidates'
                continue

            train_and_test = train_tasks[query]

            random.shuffle(train_and_test)

            all_test_triples = train_and_test

            if len(all_test_triples) == 0:
                continue
            if len(all_test_triples) < this_query_sample:
                query_triples = [random.choice(all_test_triples) for _ in range(this_query_sample)]
            else:
                query_triples = random.sample(all_test_triples, this_query_sample)

            query_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            query_left += [ent2id[triple[0]] for triple in query_triples]
            query_right += [ent2id[triple[2]] for triple in query_triples]

            # generate negative samples
            false_pairs_ = []
            false_left_ = []
            false_right_ = []
            for triple in query_triples:
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)
                    if noise in ent2id.keys(): # ent2id.has_key(noise):
                        if (noise not in e1rel_e2[e_h+rel]) and noise != e_t:
                            break
                false_pairs_.append([symbol2id[e_h], symbol2id[noise]])
                false_left_.append(ent2id[e_h])
                false_right_.append(ent2id[noise])

            false_pairs += false_pairs_
            false_left += false_left_
            false_right += false_right_

            rel_batch += [rel2id[query] for _ in range(this_query_sample)]

            labels += [rela2label[query]] * this_query_sample
            if count_sample_all == batch_size:
                break
        D_descriptions = rela_matrix[rel_batch]
        shuffle_ix = np.random.permutation(np.arange(len(D_descriptions)))
        D_descriptions = D_descriptions[shuffle_ix]
        query_pairs = np.array(query_pairs)[shuffle_ix]
        query_left = np.array(query_left)[shuffle_ix]
        query_right = np.array(query_right)[shuffle_ix]
        false_pairs = np.array(false_pairs)[shuffle_ix]
        false_left = np.array(false_left)[shuffle_ix]
        false_right = np.array(false_right)[shuffle_ix]
        labels = np.array(labels)[shuffle_ix]
        query_pairs = list(query_pairs)
        query_left = list(query_left)
        query_right = list(query_right)
        false_pairs = list(false_pairs)
        false_left = list(false_left)
        false_right = list(false_right)
        # yield rela_matrix[rel_batch], query_pairs, query_left, query_right, false_pairs, false_left, false_right, labels
        yield  query_pairs, query_left, query_right, labels
def train_generate_decription_GZSL(dataset, batch_size, symbol2id, ent2id, e1rel_e2, rel2id, args, rela2label, rela_matrix):
    print('##LOADING TRAINING DATA')
    # GZSL_data_path = '../../KGC_data/NELL/GZSL_data/gzsl_train_data.json'
    GZSL_data_path = os.path.join(dataset,"GZSL_data/gzsl_train_data.json")
    train_tasks = json.load(open(GZSL_data_path))   # train datasets
    print('##LOADING CANDIDATES')
    rel2candidates = json.load(open(os.path.join(dataset ,'rel2candidates_all.json')))  # candidates
    task_pool = list(train_tasks.keys())
    while True:
        count_sample_all = 0
        rel_batch, query_pairs, query_left, query_right, false_pairs, false_left, false_right, labels = [], [], [], [], [], [], [], []

        random.shuffle(task_pool)
        for query in task_pool:
            this_query_sample = random.choice([2, 3, 4, 5])
            if count_sample_all + this_query_sample > batch_size:
                this_query_sample = batch_size - count_sample_all
            count_sample_all += this_query_sample
            candidates = rel2candidates[query]
            if len(candidates) <= 20:
                # print 'not enough candidates'
                continue

            train_and_test = train_tasks[query]   # 'concept:athlete:felix_jones' ...

            random.shuffle(train_and_test)

            all_test_triples = train_and_test     # 'concept:athlete:felix_jones' ...

            if len(all_test_triples) == 0:
                continue
            if len(all_test_triples) < this_query_sample:
                query_triples = [random.choice(all_test_triples) for _ in range(this_query_sample)]
            else:
                query_triples = random.sample(all_test_triples, this_query_sample)

            query_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            query_left += [ent2id[triple[0]] for triple in query_triples]
            # query_relation += [ent2id[triple[1]] for triple in query_triples]
            query_right += [ent2id[triple[2]] for triple in query_triples]

            # generate negative samples
            false_pairs_ = []
            false_left_ = []
            false_right_ = []
            for triple in query_triples:
                e_h = triple[0]     # entity_head
                rel = triple[1]
                e_t = triple[2]     # entity_tail
                while True:
                    noise = random.choice(candidates)
                    if noise in ent2id.keys():  # ent2id.has_key(noise):
                        if (noise not in e1rel_e2[e_h + rel]) and noise != e_t:
                            break
                false_pairs_.append([symbol2id[e_h], symbol2id[noise]])
                false_left_.append(ent2id[e_h])
                false_right_.append(ent2id[noise])

            false_pairs += false_pairs_
            false_left += false_left_
            false_right += false_right_

            rel_batch += [rel2id[query] for _ in range(this_query_sample)]   # add right relationId to the list

            labels += [rela2label[query]] * this_query_sample   # add right labels to the list
            if count_sample_all == batch_size:
                break
        D_descriptions = rela_matrix[rel_batch]
        shuffle_ix = np.random.permutation(np.arange(len(D_descriptions)))
        D_descriptions = D_descriptions[shuffle_ix]
        query_pairs = np.array(query_pairs)[shuffle_ix]
        query_left = np.array(query_left)[shuffle_ix]
        query_right = np.array(query_right)[shuffle_ix]
        false_pairs = np.array(false_pairs)[shuffle_ix]
        false_left = np.array(false_left)[shuffle_ix]
        false_right = np.array(false_right)[shuffle_ix]
        labels = np.array(labels)[shuffle_ix]
        query_pairs = list(query_pairs)
        query_left = list(query_left)
        query_right = list(query_right)
        false_pairs = list(false_pairs)
        false_left = list(false_left)
        false_right = list(false_right)
        yield  query_pairs, query_left, query_right, labels

def train_generate_decription_WiKi(dataset, batch_size, symbol2id, ent2id, e1rel_e2, rel2id, args, rela2label, rela_matrix): # 二delete
    print('##LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    print('##LOADING CANDIDATES')
    use_less_key = dataset + '/relation_filter.txt'
    useless_kepool = []
    with open(use_less_key,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            useless_kepool.append(line)
    task_pool = list(train_tasks.keys())
    print(useless_kepool)
    while True:
        random.shuffle(task_pool)
        count_sample_all = 0
        rel_batch, query_pairs, query_left, query_right, false_pairs, false_left, false_right, labels = [], [], [], [], [], [], [], []
        for query in task_pool:
            if query in useless_kepool:
                continue
            this_query_sample = random.choice([2,2,2,3,3,4,5])
            if count_sample_all + this_query_sample > batch_size:
                this_query_sample = batch_size - count_sample_all
            count_sample_all += this_query_sample
            train_and_test = train_tasks[query]
            random.shuffle(train_and_test)
            all_test_triples = train_and_test
            if len(all_test_triples) == 0:
                continue
            if len(all_test_triples) < this_query_sample:
                query_triples = [random.choice(all_test_triples) for _ in range(this_query_sample)]
            else:
                query_triples = random.sample(all_test_triples, this_query_sample)

            query_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            query_left += [ent2id[triple[0]] for triple in query_triples]
            query_right += [ent2id[triple[2]] for triple in query_triples]
            # generate negative samples
            rel_batch += [rel2id[query] for _ in range(this_query_sample)]

            labels += [rela2label[query]] * this_query_sample
            if count_sample_all == batch_size:
                break
        D_descriptions = rela_matrix[rel_batch]
        shuffle_ix = np.random.permutation(np.arange(len(D_descriptions)))
        D_descriptions = D_descriptions[shuffle_ix]
        query_pairs = np.array(query_pairs)[shuffle_ix]
        query_left = np.array(query_left)[shuffle_ix]
        query_right = np.array(query_right)[shuffle_ix]
        labels = np.array(labels)[shuffle_ix]
        query_pairs = list(query_pairs)
        query_left = list(query_left)
        query_right = list(query_right)

        yield  query_pairs, query_left, query_right, labels

def train_generate_decription_WiKi_GZSL(dataset, batch_size, symbol2id, ent2id, e1rel_e2, rel2id, args, rela2label, rela_matrix):

    GZSL_data_path = os.path.join(dataset,'GZSL_data/gzsl_train_data.json')
    train_tasks = json.load(open(GZSL_data_path))
    print('##LOADING CANDIDATES')
    # use_less_key = './KGC_data/relation_filter.txt'
    use_less_key = os.path.join(dataset,'relation_filter.txt')
    useless_kepool = []
    with open(use_less_key, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            useless_kepool.append(line)
    print(useless_kepool)
    task_pool = list(train_tasks.keys())
    while True:
        random.shuffle(task_pool)
        count_sample_all = 0
        rel_batch, query_pairs, query_left, query_right, false_pairs, false_left, false_right, labels = [], [], [], [], [], [], [], []
        for query in task_pool:
            if query in useless_kepool:
                continue
            this_query_sample = random.choice([ 3,3,3,4,4,5])
            if count_sample_all + this_query_sample > batch_size:
                this_query_sample = batch_size - count_sample_all
            count_sample_all += this_query_sample
            train_and_test = train_tasks[query]
            random.shuffle(train_and_test)
            all_test_triples = train_and_test
            if len(all_test_triples) == 0:
                continue
            if len(all_test_triples) < this_query_sample:
                query_triples = [random.choice(all_test_triples) for _ in range(this_query_sample)]
            else:
                query_triples = random.sample(all_test_triples, this_query_sample)

            query_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            query_left += [ent2id[triple[0]] for triple in query_triples]
            query_right += [ent2id[triple[2]] for triple in query_triples]
            # generate negative samples
            rel_batch += [rel2id[query] for _ in range(this_query_sample)]

            labels += [rela2label[query]] * this_query_sample
            if count_sample_all == batch_size:
                break
        D_descriptions = rela_matrix[rel_batch]
        shuffle_ix = np.random.permutation(np.arange(len(D_descriptions)))
        D_descriptions = D_descriptions[shuffle_ix]
        query_pairs = np.array(query_pairs)[shuffle_ix]
        query_left = np.array(query_left)[shuffle_ix]
        query_right = np.array(query_right)[shuffle_ix]
        labels = np.array(labels)[shuffle_ix]
        query_pairs = list(query_pairs)
        query_left = list(query_left)
        query_right = list(query_right)

        yield  query_pairs, query_left, query_right, labels

def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:break
    return item