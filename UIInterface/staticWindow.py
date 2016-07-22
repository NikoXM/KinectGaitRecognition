import sys
import os
from PyQt4 import QtGui
from PyQt4 import QtCore
from Filter import Filter
from Window import Window
from StaticAnalyzer import StaticAnalyzer
from RandomSelector import RandomSelector
from DynamicAnalyzer import DynamicAnalyzer
from StaticAnalyzer import StaticAnalyzer
from Classifier import Classifier

limbDescriptors = ['neck','rshoulder','lshoulder','rarm','larm','rfarm','lfarm',
					'rhand','lhand','uspine','lspine','rhip','lhip','rthigh','lthigh',
					'rcalf','lcalf','rfoot','lfoot','height']

class StaticWindow(Window):
    def __init__(self):
        super(StaticWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.selection = {}
        for i in limbDescriptors:
            self.selection[i] = 0

        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Select Static Parameter')

        self.pbar = QtGui.QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 25)

        self.timer = QtCore.QBasicTimer()
        self.step = 0
        
        self.rshoulder = QtGui.QCheckBox('Right Shoulder', self)
        self.rshoulder.move(30,100)
        self.lshoulder = QtGui.QCheckBox('Left Shoulder', self)
        self.lshoulder.move(300,100)
        self.rarm = QtGui.QCheckBox('Right Arm', self)
        self.rarm.move(30,150)
        self.larm = QtGui.QCheckBox('Left Arm', self)
        self.larm.move(300,150)
        self.rfarm = QtGui.QCheckBox('Right Front Arm', self)
        self.rfarm.move(30,200)
        self.lfarm = QtGui.QCheckBox('Left Front Arm', self)
        self.lfarm.move(300,200)
        self.rhand = QtGui.QCheckBox('Right Hand', self)
        self.rhand.move(30,250)
        self.lhand = QtGui.QCheckBox('Left Hand', self)
        self.lhand.move(300,250)
        self.uspine = QtGui.QCheckBox('Upper Spine', self)
        self.uspine.move(30,300)
        self.lspine = QtGui.QCheckBox('Lower Spine', self)
        self.lspine.move(300,300)
        self.rhip = QtGui.QCheckBox('Right Hip', self)
        self.rhip.move(30,350)
        self.lhip = QtGui.QCheckBox('Left Hip', self)
        self.lhip.move(300,350)
        self.rthigh = QtGui.QCheckBox('Right Thigh', self)
        self.rthigh.move(30,400)
        self.lthigh = QtGui.QCheckBox('Left Thigh', self)
        self.lthigh.move(300,400)
        self.rcalf = QtGui.QCheckBox('Right Calf', self)
        self.rcalf.move(30,450)
        self.lcalf = QtGui.QCheckBox('Left Calf', self)
        self.lcalf.move(300,450)
        self.rfoot = QtGui.QCheckBox('Right Foot', self)
        self.rfoot.move(30,500)
        self.lfoot = QtGui.QCheckBox('Left Foot', self)
        self.lfoot.move(300,500)
        self.neck = QtGui.QCheckBox('Neck', self)
        self.neck.move(30,550)
        self.height = QtGui.QCheckBox('Height', self)
        self.height.move(300,550)
        
        self.neck.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.connect(self.rarm, QtCore.SIGNAL('stateChanged(int)'),self.rarmPush)
        self.connect(self.larm, QtCore.SIGNAL('stateChanged(int)'),self.larmPush)
        self.connect(self.rfarm, QtCore.SIGNAL('stateChanged(int)'),self.rfarmPush)
        self.connect(self.lfarm, QtCore.SIGNAL('stateChanged(int)'),self.lfarmPush)
        self.connect(self.rshoulder, QtCore.SIGNAL('stateChanged(int)'),self.rshoulderPush)
        self.connect(self.lshoulder, QtCore.SIGNAL('stateChanged(int)'),self.lshoulderPush)
        self.connect(self.rhand, QtCore.SIGNAL('stateChanged(int)'),self.rhandPush)
        self.connect(self.lhand, QtCore.SIGNAL('stateChanged(int)'),self.lhandPush)
        self.connect(self.uspine, QtCore.SIGNAL('stateChanged(int)'),self.uspinePush)
        self.connect(self.lspine, QtCore.SIGNAL('stateChanged(int)'),self.lspinePush)
        self.connect(self.rhip, QtCore.SIGNAL('stateChanged(int)'),self.rhipPush)
        self.connect(self.lhip, QtCore.SIGNAL('stateChanged(int)'),self.lhipPush)
        self.connect(self.rthigh, QtCore.SIGNAL('stateChanged(int)'),self.rthighPush)
        self.connect(self.lthigh, QtCore.SIGNAL('stateChanged(int)'),self.lthighPush)
        self.connect(self.rcalf, QtCore.SIGNAL('stateChanged(int)'),self.rcalfPush)
        self.connect(self.lcalf, QtCore.SIGNAL('stateChanged(int)'),self.lcalfPush)
        self.connect(self.rfoot, QtCore.SIGNAL('stateChanged(int)'),self.rfootPush)
        self.connect(self.lfoot, QtCore.SIGNAL('stateChanged(int)'),self.lfootPush)
        self.connect(self.neck, QtCore.SIGNAL('stateChanged(int)'),self.neckPush)
        self.connect(self.height, QtCore.SIGNAL('stateChanged(int)'),self.heightPush)

        self.confirmButton = QtGui.QPushButton("Confirm",self)
        self.confirmButton.resize(200,80)
        self.confirmButton.move(30,650)

        self.cancelButton = QtGui.QPushButton("cancel",self)
        self.cancelButton.resize(200,80)
        self.cancelButton.move(300,650)

        self.connect(self.confirmButton,QtCore.SIGNAL('clicked()'),self.confirmPush)
        self.connect(self.cancelButton,QtCore.SIGNAL('clicked()'), QtCore.SLOT('close()'))
        self.resize(550,800)
        self.text = "Please select the limbs:"

    def rarmPush(self,value):
        if self.rarm.isChecked():
            self.selection["rarm"] = 1
        else:
            self.selection["rarm"] = 0
    def larmPush(self,value):
        if self.larm.isChecked():
            self.selection["larm"] = 1
        else:
            self.selection["larm"] = 0
    def rshoulderPush(self,value):
        if self.rshoulder.isChecked():
            self.selection["rshoulder"] = 1
        else:
            self.selection["rshoulder"] = 0
    def lshoulderPush(self,value):
        if self.lshoulder.isChecked():
            self.selection["lshoulder"] = 1
        else:
            self.selection["lshoulder"] = 0
    def rfarmPush(self,value):
        if self.rfarm.isChecked():
            self.selection["rfarm"] = 1
        else:
            self.selection["rfarm"] = 0
    def lfarmPush(self,value):
        if self.lfarm.isChecked():
            self.selection["lfarm"] = 1
        else:
            self.selection["lfarm"] = 0
    def rhandPush(self,value):
        if self.rhand.isChecked():
            self.selection["rhand"] = 1
        else:
            self.selection["rhand"] = 0
    def lhandPush(self,value):
        if self.lhand.isChecked():
            self.selection["lhand"] = 1
        else:
            self.selection["lhand"] = 0
    def uspinePush(self,value):
        if self.uspine.isChecked():
            self.selection["uspine"] = 1
        else:
            self.selection["uspine"] = 0
    def lspinePush(self,value):
        if self.lspine.isChecked():
            self.selection["lspine"] = 1
        else:
            self.selection["lspine"] = 0
    def rhipPush(self,value):
        if self.rhip.isChecked():
            self.selection["rhip"] = 1
        else:
            self.selection["rhip"] = 0
    def lhipPush(self,value):
        if self.lhip.isChecked():
            self.selection["lhip"] = 1
        else:
            self.selection["lhip"] = 0
    def rthighPush(self,value):
        if self.rthigh.isChecked():
            self.selection["rthigh"] = 1
        else:
            self.selection["rthigh"] = 0
    def lthighPush(self,value):
        if self.lthigh.isChecked():
            self.selection["lthigh"] = 1
        else:
            self.selection["lthigh"] = 0
    def rcalfPush(self,value):
        if self.rcalf.isChecked():
            self.selection["rcalf"] = 1
        else:
            self.selection["rcalf"] = 0
    def lcalfPush(self,value):
        if self.lcalf.isChecked():
            self.selection["lcalf"] = 0
        else:
            self.selection["lcalf"] = 1
    def rfootPush(self,value):
        if self.rfoot.isChecked():
            self.selection["rfoot"] = 1
        else:
            self.selection["rfoot"] = 0
    def lfootPush(self,value):
        if self.lfoot.isChecked():
            self.selection["lfoot"] = 1
        else:
            self.selection["lfoot"] = 0
    def neckPush(self,value):
        if self.neck.isChecked():
            self.selection["neck"] = 1
        else:
            self.selection["neck"] = 0
    def heightPush(self,value):
        if self.height.isChecked():
            self.selection["height"] = 1
        else:
            self.selection["height"] = 0
    
    def confirmPush(self):
        limbList = []
        for p in self.selection:
            if self.selection[p] == 1:
                limbList.append(p)
        self.pbar.setValue(0)
        homedir = os.getcwd()
        filt = Filter(homedir)
        filt.dataProcess()
        self.pbar.setValue(25)
        select = RandomSelector(homedir)
        select.dataProcess()
        self.pbar.setValue(50)
        st = StaticAnalyzer(homedir,limbList)
        st.dataProcess()
        self.pbar.setValue(75)
        c = Classifier(homedir)
        count,rate,total,result = c.staticClassify()
        self.pbar.setValue(100)
        reply = QtGui.QMessageBox.question(self, 'Static Analysis Result',"Total number is %d"%(total)+"\nCorrect number is %d"%(count)+"\nCorrect rate is %f"%(100*rate)+"%", QtGui.QMessageBox.Yes)
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = SaticWindow()
    ex.show()
    app.exec_()