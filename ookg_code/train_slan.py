import openke
from openke.config import Trainer, Tester
from openke.module.model import SLAN
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from openke.data.Support import load_supports_from_txt, read_entity
import argparse
from openke.data.SimRank import load_neigh_from_txt
import numpy as np
import os
import os.path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# dataloader for training

parser = argparse.ArgumentParser(
    description="SLAN"
)

parser.add_argument(
    '--max_epochs', default=2000, type=int,
    help="Number of epochs."
)

parser.add_argument(
    '--nei_num', default=16, type=int,
    help="Number of neighbors."
)

parser.add_argument(
    '--dim', default=100, type=int,
    help="Dimension of embedding."
)

parser.add_argument(
    '--margin', default=6.0, type=int,
    help="Margin value."
)

parser.add_argument(
    '--loss_weight', default=0.2, type=int,
    help="loss weight."
)

parser.add_argument(
    '--rate', default=5.0, type=int,
    help="Learning rate."
)

parser.add_argument(
    '--save_path', default='./checkpoint/transe1', type=str,
    help="Margin value."
)

args = parser.parse_args()


train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237_10/",
	nbatches = 128,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 5,
	neg_rel = 0)
    

if os.path.isfile('testfb/nei_ent_s.npy'):
    nei_ent_s=np.load('testfb/nei_ent_s.npy')
    nei_rel_s=np.load('testfb/nei_rel_s.npy')
    as_mask=np.load('testfb/as_mask.npy')
    nei_ent_o=np.load('testfb/nei_ent_o.npy')
    nei_rel_o=np.load('testfb/nei_rel_o.npy')
    ao_mask=np.load('testfb/ao_mask.npy')
    sim_train=np.load('testfb/e_nei_sim_train.npy')
    sim_mask_train=np.load('testfb/e_weight_sim_train.npy')
else:
    ent= read_entity('./benchmarks/FB15K237_10/entity_ind.txt')
    nei_ent_s, nei_rel_s, as_mask, nei_ent_o,nei_rel_o, ao_mask= load_supports_from_txt('./benchmarks/FB15K237_10/train2id.txt', './benchmarks/FB15K237_10/auxiliary_ind.txt', ent,
                                            train_dataloader.get_ent_tot(), train_dataloader.get_rel_tot(), args.nei_num, padding=True)

    sim_train, sim_mask_train=load_neigh_from_txt('./benchmarks/FB15K237_10/train2id.txt', './benchmarks/FB15K237_10/auxiliary_ind.txt', ent,
                                            train_dataloader.get_ent_tot(), train_dataloader.get_rel_tot(), args.nei_num, 'train')




for i in range(train_dataloader.get_ent_tot()):                            
    if as_mask[i][0]==0 and ao_mask[i][0]==0:          
        ao_mask[i][0]=1 

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237_10/", "link")

# define the model
transe = SLAN(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = args.dim,
	p_norm = 1, 
	norm_flag = False,
	nei_num = args.nei_num,
    nei_ent_s=nei_ent_s, nei_rel_s=nei_rel_s, nei_ent_o=nei_ent_o, nei_rel_o=nei_rel_o, as_mask=as_mask, ao_mask=ao_mask, sim_ind=sim_train, sim_val=sim_mask_train
    )

# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = args.margin, lam = args.loss_weight),
	batch_size = train_dataloader.get_batch_size(),
)

trainer = Trainer(model = model, data_loader = train_dataloader, train_times = args.max_epochs, alpha = args.rate, use_gpu = True)
trainer.run()
transe.save_checkpoint(args.save_path+'.ckpt')


if os.path.isfile('testfb/e_nei_sim_test.npy'):
    sim_test=np.load('testfb/e_nei_sim_test.npy')
    sim_mask_test=np.load('testfb/e_weight_sim_test.npy')
else:
    ent= read_entity('./benchmarks/FB15K237_10/entity_ind.txt')
    sim_test, sim_mask_test=load_neigh_from_txt('./benchmarks/FB15K237_10/train2id.txt', './benchmarks/FB15K237_10/auxiliary_ind.txt', ent,
                                            train_dataloader.get_ent_tot(), train_dataloader.get_rel_tot(), args.nei_num, 'test')
			
transe.set_parameter(sim_ind=sim_test, sim_val=sim_mask_test)

# test the model
transe.load_checkpoint(args.save_path+'.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
