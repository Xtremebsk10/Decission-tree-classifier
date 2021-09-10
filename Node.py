

class Node:

	def __init__(self, v):
		self.name = ''
		self._class = ''
		self.root = False
		self.leave = False
		self.parent = None
		self.children = []
		self.attr_value = ''
		self.prbs={}

		for indice in v:
			self.prbs.update({indice:0})


	def __str__(self, level=0):
		if self.root == False:
			ret = "\t"*level+repr(self.name)+" Father: ("+ repr(self.parent.name)+ ","+repr(self.attr_value)+") Class:"+repr(self._class)+"\n"
		else:
			ret = "\t"*level+repr(self.name)+" Class:"+repr(self._class)+"\n"
		for child in self.children:
			ret += child.__str__(level+1)
		return ret

	def run(self,example,p):

		if self.leave == False:

			for i in self.children:

				if i.attr_value==example[self.name]:
					pr=i.run(example,p)
					return pr

			return self.prbs
		
		else:

			for at in self.prbs:
				if at in p:
					p[at]=(p[at]+self.prbs[at])
				else:
					p[at]=self.prbs[at]
			return p

	def search(self,example):
		prs={}
		
		for i in self.children:
			if i.attr_value==example[self.name]:
				prs=i.run(example,prs)
				return prs
		
		prs=self.prbs
		return prs
