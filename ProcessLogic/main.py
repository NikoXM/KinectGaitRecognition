import armaFilter as af
import dynamicAnalysis as da
import staticAnalysis as sa
import classifier as cl
import randomSelect as rs
import os

if __name__ == "__main__":
	homedir = os.getcwd()
	print homedir
	filt = af.Filter(homedir)
	filt.data_process()
	select = rs.RandomSelect(homedir)
	select.data_process()
	dy = da.DynamicAnalyzer(homedir)
	dy.data_process()
	st = sa.StaticAnalyzer(homedir)
	st.data_process()
	c = cl.Classifier(homedir)
	c.data_process()