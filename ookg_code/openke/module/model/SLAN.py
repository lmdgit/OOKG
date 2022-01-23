import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
import ipdb
import numpy as np
def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param
def to_var(x):
    return Variable(torch.from_numpy(x).cuda())

class SLAN(Model):

	def __init__(self, ent_tot, rel_tot, dim=100, p_norm=1, norm_flag=True, margin=None, epsilon=None, nei_num=None, nei_ent_s=None, nei_rel_s=None, nei_ent_o=None, nei_rel_o=None, as_mask=None, ao_mask=None, sim_ind=None, sim_val=None):
		super(SLAN, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.nei_num=nei_num
		self.nei_e_s=to_var(nei_ent_s)
		self.nei_r_s=to_var(nei_rel_s)
		self.as_mask=to_var(as_mask)
		self.nei_e_o=to_var(nei_ent_o)
		self.nei_r_o=to_var(nei_rel_o)
		self.ao_mask=to_var(ao_mask)
		self.sim_ind=to_var(sim_ind)
		self.sim_val=to_var(sim_val)
        
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot+1, self.dim)
		self.W_s=get_param((self.dim, 1 ))
		self.W_o=get_param((self.dim, 1 ))	
		self.W_e=get_param((self.dim, 1 ))         

		self.rel_embeddings1 = nn.Embedding(self.rel_tot+1, self.dim)
      
        

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings1.weight.data)




	def set_parameter(self, sim_ind=None, sim_val=None):
		self.sim_ind=to_var(sim_ind)
		self.sim_val=to_var(sim_val)    
          
       



	def _aggregation(self, nei_es, nei_rs, nei_eo, nei_ro, as_mask, ao_mask, sim_ind, sim_val, query):
		W_s=self.W_s
		W_o=self.W_o
		W_e=self.W_e
		#ipdb.set_trace()
		nei_e=nei_es.view(nei_es.shape[0]*self.nei_num)
		nei_r=nei_rs.view(nei_rs.shape[0]*self.nei_num)
		sim_ind=sim_ind.view(sim_ind.shape[0]*self.nei_num)        
		h_as = self.ent_embeddings(nei_e)
		r_as = self.rel_embeddings(nei_r)
		att_s = self.rel_embeddings1(nei_r)        
		h_as = h_as - r_as 		
		h_as=h_as.view(as_mask.shape[0], self.nei_num, -1)
		hs_all=h_as.transpose(1,2)
        
		h_e = self.ent_embeddings(sim_ind)
		h_e=h_e.view(as_mask.shape[0], self.nei_num, -1)
		he_all=h_e.transpose(1,2)       
		att_s=att_s.view(as_mask.shape[0], self.nei_num, -1)* query.unsqueeze(dim=1)
		atts_all = torch.matmul(att_s, W_s)        
		atts_all=atts_all.transpose(1,2).squeeze()              
		atte_all = torch.mm(query , W_e)              
 
		nei_eo=nei_eo.view(nei_eo.shape[0]*self.nei_num)
		nei_ro=nei_ro.view(nei_ro.shape[0]*self.nei_num)
		h_ao = self.ent_embeddings(nei_eo)
		r_ao = self.rel_embeddings(nei_ro)
		h_ao = h_ao + r_ao 		
		h_ao=h_ao.view(as_mask.shape[0], self.nei_num, -1)
		ho_all=h_ao.transpose(1,2)
        
		att_o = self.rel_embeddings1(nei_ro)
		att_o=att_o.view(as_mask.shape[0], self.nei_num, -1)* query.unsqueeze(dim=1)
		atto_all = torch.matmul(att_o, W_o)        
		atto_all=atto_all.transpose(1,2).squeeze()

		atts_all= abs(atts_all)
		atto_all= abs(atto_all)
		atte_all= abs(atte_all)

        
		a_mask=torch.cat([as_mask * atts_all, ao_mask * atto_all, sim_val * atte_all], dim=-1)
		h_all=torch.cat([hs_all, ho_all, he_all], dim=-1)

        
		a_mask=a_mask.unsqueeze(dim=1)        
		h_all=h_all * a_mask
		a_mask = a_mask.sum(dim=-1)
		h_o = h_all.sum(dim=-1)
		h_o = h_o / a_mask
        
		return h_o

	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def forward(self, data, mode='normal'):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
      
		mode = data['mode']
		nei_r_s= self.nei_r_s
		nei_e_s = self.nei_e_s

		as_mask = self.as_mask
		nei_r_o= self.nei_r_o
		nei_e_o = self.nei_e_o
		ao_mask = self.ao_mask
		sim_ind = self.sim_ind
		sim_val = self.sim_val   
      
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		query = self.rel_embeddings1(batch_r)
         
		h_o= self._aggregation(nei_e_s[batch_h,:], nei_r_s[batch_h,:], nei_e_o[batch_h,:], nei_r_o[batch_h,:], as_mask[batch_h,:], ao_mask[batch_h,:], sim_ind[batch_h,:], sim_val[batch_h,:], query)

		t_o= self._aggregation(nei_e_s[batch_t,:], nei_r_s[batch_t,:], nei_e_o[batch_t,:], nei_r_o[batch_t,:], as_mask[batch_t,:], ao_mask[batch_t,:], sim_ind[batch_t,:], sim_val[batch_t,:], query)
    
        
		
		score1 = self._calc(h, t, r, mode)
		score2 = self._calc(h_o, t, r, mode)
		score3 = self._calc(h, t_o, r, mode)        

        

		return score1, score2, score3

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score1,score2,score3= self.forward(data)
		mode = data['mode']
		if mode == 'head_batch':
			score = score3
		else:
			score = score2

		return score.cpu().data.numpy() 