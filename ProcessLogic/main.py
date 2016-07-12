from ArmaFilter import Filter
from DynamicAnalyzer import DynamicAnalyzer
from StaticAnalyzer import StaticAnalyzer
from Classifier import Classifier
from RandomSelect import RandomSelect
import os
import sys

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
	select = RandomSelect(homedir)
	select.data_process()
	dy = DynamicAnalyzer(homedir)
	dy.data_process()
	st = StaticAnalyzer(homedir)
	st.data_process()
	c = Classifier(homedir)
	c.data_process()