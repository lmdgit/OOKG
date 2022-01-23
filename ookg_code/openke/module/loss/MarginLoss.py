import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss
import ipdb
class MarginLoss(Loss):

	def __init__(self, adv_temperature = None, margin = 6.0, lam = 0.2):
		super(MarginLoss, self).__init__()
		#ipdb.set_trace()
		self.margin = nn.Parameter(torch.Tensor([margin])) 
		self.lam = nn.Parameter(torch.Tensor([lam]))
		self.lam.requires_grad = False		
		self.margin.requires_grad = False
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score1, n_score1, p_score2, n_score2, p_score3, n_score3):
		if self.adv_flag:
			return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
		else:
			loss1=(torch.max(p_score1 - n_score1, -self.margin)).mean() + self.margin
			loss2= (torch.max(p_score2 - n_score2, -self.margin)).mean() + self.margin
			loss3= (torch.max(p_score3 - n_score3, -self.margin)).mean() + self.margin            
			return loss1 + self.lam * loss2 + self.lam * loss3
			
	
	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()