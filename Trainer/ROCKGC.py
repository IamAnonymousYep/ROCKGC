import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.optim as optim
import argparse
import random
import math
from time import gmtime, strftime
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import *
from dataset_GBU import  DATA_LOADER
from utils import *
from create_train_feature import Feature_extracter_addloss
from data_loader import train_generate_decription_GZSL, train_generate_decription, train_generate_decription_WiKi, train_generate_decription_WiKi_GZSL

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="GZSL", help='ZSL or GZSL')
parser.add_argument('--dataset', default='NELL',help='dataset: NELL,WiKi')
parser.add_argument('--embed_model', default='DistMult', help='DistMult, TransE')

###################################################################

parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--gen_nepoch', type=int, default=50, help='number of epochs to train for')   # initial equals 50
parser.add_argument('--lr', type=float, default=0.00003, help='learning rate to train generater')  # initial equals 0.00003
###################################################################

parser.add_argument('--beta', type=float, default=1, help='tc weight')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--dis', type=float, default=0.3, help='Discriminator weight')
parser.add_argument('--dis_step', type=float, default=2, help='Discriminator update interval')
parser.add_argument('--kl_warmup', type=float, default=0.005, help='kl warm-up for VAE') # initial is 0.005
parser.add_argument('--tc_warmup', type=float, default=0.005, help='tc warm-up')

parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')
parser.add_argument('--ae_drop', type=float, default=0.2, help='dropout rate in the auto-encoder')

parser.add_argument('--batchsize', type=int, default=64, help='input batch size')   # initial equals 64
parser.add_argument('--nSample', type=int, default=20, help='number features to generate per class')

parser.add_argument('--disp_interval', type=int, default=1000)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=5000) # initial equals 5000
parser.add_argument('--evl_start',  type=int, default=29999) # initial equals 29999
parser.add_argument('--manualSeed', type=int, default=6152, help='manual seed')  # 6152

# Paramater of trained Feature Encoder

parser.add_argument('--entity_dim', type=int, default=50)   # initial equals float
parser.add_argument('--embed_dim', type=int, default=50)    # initial equals float
parser.add_argument('--latent_dim_size', type=float, default=2)
parser.add_argument('--pretrain_margin', default=10.0, type=float, help='margin of Feature Encoder')
parser.add_argument('--add_loss_weight', default=0.01, type=float, help='class loss weight of Feature Encoder')
parser.add_argument('--ep_dim', type=int, default=800)
parser.add_argument('--extractor_way', default='_addloss', help='extractor_embed_type')
parser.add_argument('--sp_dim1', default=800, type=int)
parser.add_argument('--sp_dim2', default=800, type=int)


# Paramater of VAE45
parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
parser.add_argument('--vae_encoder_dim', type=int, default=800)
parser.add_argument('--vae_decoder_dim', type=int, default=800)
# Paramater of Denoising AE

parser.add_argument('--RS_dim', type=int, default=400)
parser.add_argument('--IS_dim', type=int, default=400)
parser.add_argument('--SCN_dim', type=int, default=200)

parser.add_argument('--ga', type=float, default=5, help='relationNet weight')
parser.add_argument('--ca', default=100, type=float, help='compare closs weight')
parser.add_argument('--co', default=1.5, type=float, help='compare closs weight')
parser.add_argument('--co_ratio', default=3.5, type=float, help='compare closs weight')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
opt = parser.parse_args()


# path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
opt.data_path = os.path.join(root_path,"KGC_data", opt.dataset)
opt.data_presave_path = os.path.join(root_path,"data_presave")
opt.extractor_path = os.path.join(root_path,"saved_model", opt.dataset) #path to extractor

np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))

if torch.cuda.is_available():
    opt.gpu = torch.device(f"cuda:{opt.gpu}")
else:
    opt.gpu = torch.device("cpu")
print("opt.gpu: ", opt.gpu)

def train():
    writer = SummaryWriter('runs/CZRL')
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class
    out_dir = 'out/{}/wd-{}_b-{}_g-{}_lr-{}_sd-{}_dis-{}_nS-{}_nZ-{}_bs-{}'.format(opt.dataset, opt.weight_decay,
                    opt.beta, opt.ga, opt.lr, opt.RS_dim, opt.dis, opt.nSample, opt.Z_dim, opt.batchsize)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))
    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    dataset.feature_dim = dataset.train_feature.shape[1]
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class
    opt.niter = int(dataset.ntrain/opt.batchsize) * opt.gen_nepoch
    model = VAE(opt).to(opt.gpu)
    SCN = SC_Net(opt).to(opt.gpu)
    ae = AE(opt).to(opt.gpu)
    # print(model)
    # print(ae)
    # print(SCN)
    #########################################################################
    e1rel_e2 = json.load(open(os.path.join(opt.data_path,  'e1rel_e2_all.json')))
    FE = Feature_extracter_addloss(opt)
    Ep = FE.Extractor
    Ep.cuda()
    #########################################################################
    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('Generative Model Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    relation_optimizer = optim.Adam(SCN.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ae_optimizer = optim.Adam(ae.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    mse = nn.MSELoss().to(opt.gpu)
    iters = math.ceil(dataset.ntrain/opt.batchsize)
    beta = 0.01
    model.train()
    ae.train()
    SCN.train()
    if opt.task == "GZSL" and opt.dataset == "NELL":
        G_data = train_generate_decription_GZSL(opt.data_path, opt.batchsize, FE.symbol2id, FE.ent2id,
                                       e1rel_e2, FE.rel2id, opt, FE.rel2id, FE.rela_matrix)
    elif opt.task == "ZSL" and opt.dataset == "NELL":
        G_data = train_generate_decription(opt.data_path, opt.batchsize, FE.symbol2id, FE.ent2id,
                                       e1rel_e2, FE.rel2id, opt, FE.rel2id, FE.rela_matrix)
    elif opt.task == "GZSL" and opt.dataset == "WiKi":
        G_data = train_generate_decription_WiKi_GZSL(opt.data_path, opt.batchsize, FE.symbol2id, FE.ent2id,
                                       e1rel_e2, FE.rel2id, opt, FE.rel2id, dataset.attribute)
    elif opt.task == "ZSL" and opt.dataset == "WiKi":
        G_data = train_generate_decription_WiKi(opt.data_path, opt.batchsize, FE.symbol2id, FE.ent2id,
                                       e1rel_e2, FE.rel2id, opt, FE.rel2id, dataset.attribute)
    else: assert False
    for it in range(start_step, opt.niter+1):
        if it % iters == 0:
            beta = min(opt.kl_warmup*(it/iters), 1)
        query, query_left, query_right, labels_numpy = next(G_data)
        query_meta = FE.get_meta(query_left, query_right)
        query_array = np.array(query)
        query = torch.LongTensor(query_array).cuda()
        # query = Variable(torch.LongTensor(query)).cuda()
        X_, _, _ = Ep(query, query, query_meta, query_meta)
        labels_ = torch.from_numpy(labels_numpy.astype('int')).to(opt.gpu)
        C = np.array([dataset.attribute[i, :] for i in labels_])
        C = torch.from_numpy(C.astype('float32')).to(opt.gpu)
        X = X_.cuda()
        sample_C = torch.from_numpy(np.array([dataset.attribute[i, :] for i in labels_.unique()])).to(opt.gpu)
        sample_C_n = labels_.unique().shape[0]
        sample_label = labels_.unique().cpu()
        sample_C_is = torch.from_numpy(np.array([dataset.attribute[i, :] for i in labels_])).to(opt.gpu)
        x_mean, z_mu, z_var, z = model(X, C)

        ########################################################

        sample_labels = np.array(sample_label)
        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        one_hot_labels = torch.zeros(opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1).to(opt.gpu)
        centroid_matrix = torch.zeros((sample_C_n, opt.ep_dim))
        centroid_matrix = centroid_matrix.cuda()

        i = 0
        for label in labels_.unique():
            this_label_centroid = torch.where(labels_ == label)
            this_label_centroid = x_mean[this_label_centroid]
            centroid_matrix[i] = this_label_centroid.mean(dim=0)
            i += 1
        sim = F.cosine_similarity(x_mean.unsqueeze(1), centroid_matrix.unsqueeze(0), dim=2)

        dis = 1 - sim  # 64 * 139
        Label = torch.FloatTensor(np.array(one_hot_labels.cpu().detach().numpy())).to(opt.gpu)
        Zero_tensor = torch.zeros_like(Label)
        loss_inter = torch.where(Label == 1, dis, Zero_tensor)
        loss_inter = torch.sum(loss_inter, dim=1).mean()
        dis = opt.co * torch.ones_like(dis) - dis

        dis = torch.where(dis > 0, dis, Zero_tensor)
        loss_intra = torch.where(Label != 1, dis, Zero_tensor)
        loss_intra = torch.mean(loss_intra, dim=1).sum()
        loss_cvae_compare = loss_inter + loss_intra
        loss, ce, kl = multinomial_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)
        loss_vae = loss.data.item()
        sample_labels = np.array(sample_label)
        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        one_hot_labels = torch.zeros(opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1).to(opt.gpu)

        x1, s1, rs1, is1 = ae(x_mean)
        relations = SCN(rs1, sample_C, Flag="rs")
        relations = relations.view(-1, labels_.unique().cpu().shape[0])
        p_loss = opt.ga * mse(relations, one_hot_labels)

        x2, s2, rs2, is2 = ae(X)
        relations = SCN(rs2, sample_C, Flag="rs")
        relations = relations.view(-1, labels_.unique().cpu().shape[0])
        p_loss = p_loss + opt.ga * mse(relations, one_hot_labels)

        rec = mse(x1, X) + mse(x2, X)
        one_hot_labels_is = torch.zeros(opt.batchsize).to(opt.gpu)
        relations = SCN(is1, sample_C_is, Flag="is").view(-1)
        p_loss1 = opt.ca * mse(relations, one_hot_labels_is)
        relations = SCN(is2, sample_C_is, Flag="is").view(-1)
        p_loss1 = p_loss1 + opt.ca * mse(relations, one_hot_labels_is)
        ################################################################
        writer.add_scalar('Loss/total_loss', loss.item(), it)
        writer.add_scalar('Loss/kl_loss', kl.item(), it)
        writer.add_scalar('Loss/p_loss', p_loss.item(), it)
        writer.add_scalar('Loss/rec_loss', rec.item(), it)
        writer.add_scalar('Loss/p_loss1', p_loss1.item(), it)
        writer.add_scalar('Loss/loss_cvae_compare', loss_cvae_compare.item(), it)
        ################################################################
        loss = loss + p_loss + rec + p_loss1 + opt.co_ratio * loss_cvae_compare
        optimizer.zero_grad()
        relation_optimizer.zero_grad()
        ae_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        relation_optimizer.step()
        ae_optimizer.step()
        model.eval()
        ae.eval()
        Ep.eval()
        SCN.eval()
        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; loss: {:.3f}; kl:{:.3f}; p_loss:{:.3f}; rec:{:.3f} ; p_loss1:{:.3f}; loss_cvae_compare:{:.3f}'.format(
                it,opt.niter, loss_vae, kl.item(), p_loss.item(), rec.item(), p_loss1.item(),loss_cvae_compare.item())
            log_print(log_text, log_dir)
        if it % opt.evl_interval == 0 and it > opt.evl_start:
            if opt.task == "GZSL":
                eval_relation_addloss_GZSL(model, ae, Ep, SCN, FE, dataset, opt, mode='test')
            elif opt.task == "ZSL":
                if opt.dataset == "NELL":
                    eval_relation_addloss_nell(model, ae, Ep, SCN, FE, dataset, opt, mode='test')    # initial without _nell
                elif opt.dataset == "WiKi":
                    eval_relation_addloss_wiki(model, ae, Ep, SCN, FE, dataset, opt, mode='test')
                else :
                    raise ValueError("the dataset is not in ['NELL', 'WiKi']")
            else:
                assert False
        model.train()
        ae.train()
        SCN.train()
    writer.close()

if __name__ == "__main__":
    train()
