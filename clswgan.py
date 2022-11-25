from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import classifier
import classifier2
import sys
import model
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/BS/xian/work/cvpr18-code-release/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--anti', type=float, default=0.01, help='anti')
parser.add_argument('--dec_lr', type=float, default= 0.0001, help='dec_lr')
parser.add_argument('--recons_weight', type=float, default= 0.1, help='recons_weight')
parser.add_argument('--gammaD', type=int,default= 10, help='gammaD')
parser.add_argument('--gammaG', type=int,default= 10, help='gammaG')
parser.add_argument('--loss_syn_num', type=int, default=30, help='G learning rate')
parser.add_argument('--dm_seen_weight', type=float, default=0.01, help='weight of the seen class cycle loss')
parser.add_argument('--dm_unseen_weight', type=float, default=0.01, help='weight of the unseen class cycle loss')
parser.add_argument('--dm_weight', type=float, default=0.01, help='weight of the unseen class cycle loss')
parser.add_argument('--cls_batch_size', type=int, default=5, help='G learning rate')
parser.add_argument('--cls_syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--l_gamma', type=float, default=0.95, help='gamma lower')
parser.add_argument('--u_gamma', type=float, default=0.99, help='gamma upper')
parser.add_argument('--proto_layer_sizes', type=list, default=[1024,2048], help='size of the hidden and output units in prototype learner')

parser.add_argument('--tem', type=float, default=0.04,help='temprature (Eq. 16)')
parser.add_argument('--ratio', type=float, default=1000,help='hyperparameter to control the seen-unseen prior (Sec. 4.4)')
opt = parser.parse_args()

print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

num_s=data.seenclasses.size(0)
num_u=data.unseenclasses.size(0)
num_class=num_s+num_u

"""
class prior
"""
log_p_y = torch.zeros(num_class).cuda()#class prior
p_y_seen = torch.zeros(num_s).cuda()#conditional class prior on seen class (near Eq. 14)
p_y_unseen = torch.ones(num_u).cuda() / num_u#conditional class prior on unseen class (near Eq. 14)
for i in range(p_y_seen.size(0)):
    iclass = data.seenclasses[i]
    index = data.train_label == iclass
    p_y_seen[i] = index.sum().float().cuda()
p_y_seen /= p_y_seen.sum()
log_p_y[data.seenclasses] = p_y_seen.log()
log_p_y[data.unseenclasses] = p_y_unseen.log()
"""
seen-unseen prior
"""
log_p0_Y0= torch.zeros(10).cuda()#seen-unsee prior (Eq. 11)
p0_u0 = (1 - p0_s)
log_p0_Y0[data.unseenclasses] = math.log(p0_u)

# initialize generator and discriminator
netDec = model.AttDec(opt,opt.attSize)
print(netDec)

netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()
dm_classifier = model.DomainClassifier(opt.resSize)
cnp_criterion = nn.CrossEntropyLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
noise1 = torch.FloatTensor(opt.batch_size, opt.nz)
noise2 = torch.FloatTensor(opt.batch_size, opt.nz)
one = one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)
input_res2 = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att2 = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label2 = torch.LongTensor(opt.batch_size)

input_res3 = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att3 = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label3 = torch.LongTensor(opt.batch_size)
gamma=opt.u_gamma

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netDec.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    noise1,noise2=noise1.cuda(),noise2.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()
    dm_classifier.cuda()
    cnp_criterion.cuda()
    input_res2 = input_res2.cuda()
    input_att2 = input_att2.cuda()
    input_label2 = input_label2.cuda()

    input_res3 = input_res3.cuda()
    input_att3 = input_att3.cuda()
    input_label3 = input_label3.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
optimizerDec  = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizer_dm = optim.Adam(dm_classifier.parameters(), lr=1e-4, betas=(opt.beta1, 0.999))
def anti_collapse_regularizer(z1, z2, fake_feats_z1, fake_feats_z2):
  return torch.mean(((1- cos(fake_feats_z1, fake_feats_z2)) /(1- cos(z1, z2))))

def sample2():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res2.copy_(batch_feature)
    input_att2.copy_(batch_att)
    input_label2.copy_(util.map_label(batch_label, data.seenclasses))

def sample3():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res3.copy_(batch_feature)
    input_att3.copy_(batch_att)
    input_label3.copy_(util.map_label(batch_label, data.seenclasses))

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
def generate_syn_feature_with_grad(netG, classes, attribute, num):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opt.attSize)
    syn_noise = torch.FloatTensor(nclass * num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        syn_label = syn_label.cuda()
    syn_noise.normal_(0, 1)
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_feature = netG(Variable(syn_noise), Variable(syn_att))
    return syn_feature, syn_label.cuda()

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False

for epoch in range(opt.nepoch):
    FP = 0 
    mean_lossD = 0
    mean_lossG = 0
    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        
        for p in netDec.parameters(): #unfreeze deocder
            p.requires_grad = True
        gp_sum = 0
        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with realG
            # sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            netDec.zero_grad()
            recons = netDec(input_resv)
            R_cost = opt.recons_weight*WeightedL1(recons, input_attv) 
            R_cost.backward()
            optimizerDec.step()
            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()
       
        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation
        for p in netDec.parameters(): #freeze decoder
            p.requires_grad = False
        for q in dm_classifier.parameters():
            q.requires_grad = True

        optimizer_dm.zero_grad()
        sample2()
        input_resv2 = Variable(input_res2)
        fake_unseen_feature1, fake_unseen_label1 = generate_syn_feature(netG, data.unseenclasses,data.attribute, opt.loss_syn_num)  # 每个类生成2个sample;31x2=62
        src_label_dm = torch.ones(input_label2.size()).long().cuda()
        tgt_label_dm = torch.zeros(fake_unseen_label1.size()).long().cuda()
        src_label_dm = Variable(src_label_dm)
        tgt_label_dm = Variable(tgt_label_dm)
        src_output_dm = dm_classifier(input_resv2)
        tgt_output_dm = dm_classifier(Variable(fake_unseen_feature1.cuda()))
        loss_dm_src = cnp_criterion(src_output_dm, src_label_dm)
        loss_dm_tgt = cnp_criterion(tgt_output_dm, tgt_label_dm)
        loss_dm = opt.dm_seen_weight * loss_dm_src + opt.dm_unseen_weight * loss_dm_tgt
        loss_dm.backward()
        optimizer_dm.step()

        for q in dm_classifier.parameters():
            q.requires_grad = False   

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)

        fake = netG(noisev, input_attv)
        k=max(gamma*opt.batch_size,opt.l_gamma*opt.batch_size)
        gamma=gamma*gamma
        # criticG_fake = netD(fake, input_attv)
        # criticG_fake = criticG_fake.mean()
        # G_cost = -criticG_fake
        criticG_fake = torch.topk(torch.transpose(netD(fake, input_attv), 0, 1),int(k))
        criticG_fake = criticG_fake.values.mean()
        G_cost = -criticG_fake
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))
        errk = G_cost + opt.cls_weight*c_errG
        errk.backward()
        fake = netG(noisev, input_attv)
        noise1.normal_(0, 1)
        noisev1 = Variable(noise1)
        fake1 = netG(noisev1, input_attv)
        noise2.normal_(0, 1)
        noisev2 = Variable(noise2)
        fake2 = netG(noisev2, input_attv)
        netDec.zero_grad()
        recons_fake = netDec(fake)
        # classification loss
        R_cost = WeightedL1(recons_fake, input_attv)
        anti_collapse=anti_collapse_regularizer(noise1, noise2, fake1, fake2)
        errG = (1/ anti_collapse)*opt.anti+opt.recons_weight * R_cost
        errG.backward()
        fake_unseen_feature2, fake_unseen_label2 = generate_syn_feature_with_grad(netG, data.unseenclasses,data.attribute,opt.loss_syn_num)
        sample3()
        input_resv3 = Variable(input_res3)
        feature_concat = torch.cat((input_resv3, fake_unseen_feature2), 0)
        output_dm_conf = dm_classifier(feature_concat)
        output_dm_conf = F.softmax(output_dm_conf, dim=1)
        uni_distrib = torch.FloatTensor(output_dm_conf.size()).uniform_(0, 1)
        uni_distrib = uni_distrib.cuda()
        uni_distrib = Variable(uni_distrib)
        loss_conf = -opt.dm_weight * (torch.sum(uni_distrib * torch.log(output_dm_conf))) / float(output_dm_conf.size(0))
        loss_conf.backward()

        optimizerG.step()
        optimizerDec.step()

    mean_lossG /=  data.ntrain / opt.batch_size 
    mean_lossD /=  data.ntrain / opt.batch_size 
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
              % (epoch, opt.nepoch, D_cost.data, G_cost.data, Wasserstein_D.data, c_errG.data))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # Generalized zero-shot learning
    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        # nclass = opt.nclass_all
        # cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.8, 55, opt.syn_num, True)
        # print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
    
        cls_gzsl = classifier_ZLAP.CLASSIFIER(log_p_y,log_p0_Y,opt.proto_layer_sizes, train_X,
                                                   train_Y, data,
                                                   opt.cuda, opt.classifier_lr, 0.5, 60,
                                                   opt.batch_size,
                                                   True,opt.tem)
        acu = cls_gzsl.acc_unseen.cpu().data.numpy()
        acs = cls_gzsl.acc_seen.cpu().data.numpy()
        ach = cls_gzsl.H.data.cpu().numpy()
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (acu, acs, ach))
    # Zero-shot learning
    else:

        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        cls = classifier_ZLAP.CLASSIFIER(log_p_y,log_p0_Y0,opt.proto_layer_sizes, train_X,
                                                   train_Y, data,
                                                   opt.cuda, opt.classifier_lr, 0.5, 60,
                                                   opt.batch_size,
                                                   False,opt.tem)
        acc = cls.acc
        print('unseen class accuracy= ', acc)
         
    # reset G to training mode
    netG.train()