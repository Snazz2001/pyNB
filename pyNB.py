"""
pyNB - a fast Naive Bayes text classifier
"""

from math import log
import re

# Default dict data structure that doesn't use too much memory
class MyDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return 0

# Utility functions
def simple_tokenizer(document):
	return re.findall(r'\w+', document)

def ngram_tokenizer(n):
	def tokenizer(document):
		words = re.findall(r'\w+', document)
		result = []
		for x in range(n):
			result.extend('-'.join(words[i:i+x+1]) for i in range(len(words)))
		return result
	return tokenizer

class NaiveBayes(object):
	def __init__(self, tokenizer=simple_tokenizer, laplace=1.0, bernoulli=False, use_priors=True):
		self.tokenizer = tokenizer
		self.laplace = laplace
		self.bernoulli = bernoulli
		self.use_priors = use_priors
		self.labels = set()
		self.counts = MyDict()
		self.sums = MyDict()
		self.total = 0L

	def train(self, documents, labels):
		for (document, label) in zip(documents, labels):
			tokens = self.tokenizer(document)
			if self.bernoulli:
				tokens = set(tokens)
			if len(tokens):
				self.labels.add(label)	
			for token in tokens:
				self.counts[(label, token)] += 1
				self.sums[label] += 1
				self.total += 1
		return self

	def classify(self, documents):
		return map(self.classify_document, documents)

	def validate(self, documents, labels):
		total = len(labels)
		corr = len(filter(lambda (x, y): x == y, zip(self.classify(documents), labels)))
		return 1.0 * corr / total

	def get_log_prob(self, label, token):
		return log(self.counts[(label, token)] + self.laplace) - log((1 + self.laplace) * self.sums[label])

	def classify_document(self, document):
		tokens = self.tokenizer(document)
		if self.bernoulli:
			tokens = set(tokens)
		if not self.use_priors:
			label = max(self.labels, key=lambda label: sum(self.get_log_prob(label, token) for token in tokens))
		else:
			label = max(self.labels, key=lambda label: sum(self.get_log_prob(label, token) for token in tokens) + 
				log(self.sums[label]))
		return label