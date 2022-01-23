from .Strategy import Strategy

class NegativeSampling(Strategy):

	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate

	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		score1, score2, score3= self.model(data)
		p_score1 = self._get_positive_score(score1)
		n_score1 = self._get_negative_score(score1)
		p_score2 = self._get_positive_score(score2)
		n_score2 = self._get_negative_score(score2)
		p_score3 = self._get_positive_score(score3)
		n_score3 = self._get_negative_score(score3)        
		loss_res = self.loss(p_score1, n_score1, p_score2, n_score2, p_score3, n_score3)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res