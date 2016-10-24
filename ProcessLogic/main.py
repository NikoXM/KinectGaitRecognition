from Filter import Filter
from DynamicAnalyzer import DynamicAnalyzer
from StaticAnalyzer import StaticAnalyzer
from Classifier import Classifier
from RandomSelector import RandomSelector
import os
import sys

limbDescriptors  = ['neck','rshoulder','lshoulder','rarm','larm','rfarm','lfarm',
					'rhand','lhand','uspine','lspine','rhip','lhip','rthigh','lthigh',
					'rcalf','lcalf','rfoot','lfoot','height']

if __name__ == "__main__":
	homedir = sys.path[0]
	import platform
	p = platform.platform().split('-')[0]
	if p == "Windows":
		seperator = "\\"
	elif p == "Darwin":
		seperator = "/"
	i = homedir.rfind(seperator)
	homedir = homedir[0:i]

	print homedir

	filt = Filter(homedir)
	filt.dataProcess()
	select = RandomSelector(homedir)
	select.dataProcess()

	lists = ['srkrar','srklal']
	dy = DynamicAnalyzer(homedir,lists)
	dy.dataProcess()
	st = StaticAnalyzer(homedir,limbDescriptors)
	st.dataProcess()
	c = Classifier(homedir)
	c.dataProcess()