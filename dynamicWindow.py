import sys
import os
import dynamicAnalysis as da
import classifier as cl
import armaFilter as af
import randomSelect as rs
from PyQt4 import QtGui
from PyQt4 import QtCore

angle_descriptors = ['srkrar','srklal','slkrar','slklal','hrklal','hlkrar','krhlal','klhrar','arhlkl','alhrkr']

class dynamicWindow(QtGui.QWidget):
    def __init__(self):
        super(dynamicWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.selection = {}
        for i in angle_descriptors:
            self.selection[i] = 0

        self.setWindowTitle('Select Dynamic Parameter')
        self.pbar = QtGui.QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 25)
        
        
        self.srkrar = QtGui.QCheckBox('Shoulder-Right Knee-Right Ankle-Right', self)
        self.srkrar.move(30,100)
        self.srklal = QtGui.QCheckBox('Shoulder-Right Knee-Left Ankle-Left', self)
        self.srklal.move(30,150)
        self.slkrar = QtGui.QCheckBox('Shoulder-Right Knee-Right Ankle-Right', self)
        self.slkrar.move(30,200)
        self.slklal = QtGui.QCheckBox('Shoulder-Left Knee-Left Ankle-Left', self)
        self.slklal.move(30,250)
        self.hrklal = QtGui.QCheckBox('Hip-Right Knee-Left Ankle-Left', self)
        self.hrklal.move(30,300)
        self.hlkrar = QtGui.QCheckBox('Hip-Left Knee-Right Ankle-Right', self)
        self.hlkrar.move(30,350)
        self.krhlal = QtGui.QCheckBox('Knee-Right Hip-Left Ankle-Left', self)
        self.krhlal.move(30,400)
        self.klhrar = QtGui.QCheckBox('Knee-Left Hip-Right Ankle-Right', self)
        self.klhrar.move(30,450)
        self.arhlkl = QtGui.QCheckBox('Ankle-Right Hip-Left Knee-Left', self)
        self.arhlkl.move(30,500)
        self.alhrkr = QtGui.QCheckBox('Ankle-Left Hip-Right Knee-Right', self)
        self.alhrkr.move(30,550)
        self.hcsckl = QtGui.QCheckBox('Hip-center Shoulder-Center Knee-Left', self)
        self.hcsckl.move(30,600)
        self.hcsckr = QtGui.QCheckBox('Hip-center Shoulder-Center Knee-Right', self)
        self.hcsckr.move(30,650)
        
        self.connect(self.srkrar, QtCore.SIGNAL('stateChanged(int)'),self.srkrarPush)
        self.connect(self.srklal, QtCore.SIGNAL('stateChanged(int)'),self.srklalPush)
        self.connect(self.slkrar, QtCore.SIGNAL('stateChanged(int)'),self.slkrarPush)
        self.connect(self.slklal, QtCore.SIGNAL('stateChanged(int)'),self.slklalPush)
        self.connect(self.hrklal, QtCore.SIGNAL('stateChanged(int)'),self.hrklalPush)
        self.connect(self.hlkrar, QtCore.SIGNAL('stateChanged(int)'),self.hlkrarPush)
        self.connect(self.krhlal, QtCore.SIGNAL('stateChanged(int)'),self.krhlalPush)
        self.connect(self.klhrar, QtCore.SIGNAL('stateChanged(int)'),self.klhrarPush)
        self.connect(self.arhlkl, QtCore.SIGNAL('stateChanged(int)'),self.arhlklPush)
        self.connect(self.alhrkr, QtCore.SIGNAL('stateChanged(int)'),self.alhrkrPush)
        self.connect(self.hcsckl, QtCore.SIGNAL('stateChanged(int)'),self.hcscklPush)
        self.connect(self.hcsckr, QtCore.SIGNAL('stateChanged(int)'),self.hcscklPush)

        self.confirmButton = QtGui.QPushButton("Confirm",self)
        self.confirmButton.resize(200,80)
        self.confirmButton.move(30,700)

        self.cancelButton = QtGui.QPushButton("cancel",self)
        self.cancelButton.resize(200,80)
        self.cancelButton.move(300,700)

        self.connect(self.confirmButton,QtCore.SIGNAL('clicked()'),self.confirmPush)
        self.connect(self.cancelButton,QtCore.SIGNAL('clicked()'), QtCore.SLOT('close()'))
        
        self.resize(550,800)

        self.text = "Please select the limbs:"
    def srkrarPush(self,value):
        if self.srkrar.isChecked():
            self.selection["srkrar"] = 1
        else:
            self.selection["srkrar"] = 0
    def srklalPush(self,value):
        if self.srklal.isChecked():
            self.selection["srklal"] = 1
        else:
            self.selection["srklal"] = 0
    def slkrarPush(self,value):
        if self.slkrar.isChecked():
            self.selection["slkrar"] = 1
        else:
            self.selection["slkrar"] = 0
    def slklalPush(self,value):
        if self.slklal.isChecked():
            self.selection["slklal"] = 1
        else:
            self.selection["slklal"] = 0
    def hrklalPush(self,value):
        if self.hrklal.isChecked():
            self.selection["hrklal"] = 1
        else:
            self.selection["hrklal"] = 0
    def hlkrarPush(self,value):
        if self.hlkrar.isChecked():
            self.selection["hlkrar"] = 1
        else:
            self.selection["hlkrar"] = 0
    def krhlalPush(self,value):
        if self.krhlal.isChecked():
            self.selection["krhlal"] = 1
        else:
            self.selection["krhlal"] = 0
    def klhrarPush(self,value):
        if self.klhrar.isChecked():
            self.selection["klhrar"] = 1
        else:
            self.selection["klhrar"] = 0
    def arhlklPush(self,value):
        if self.arhlkl.isChecked():
            self.selection["arhlkl"] = 1
        else:
            self.selection["arhlkl"] = 0
    def alhrkrPush(self,value):
        if self.alhrkr.isChecked():
            self.selection["alhrkr"] = 1
        else:
            self.selection["alhrkr"] = 0
    def hcscklPush(self,value):
        if self.hcsckl.isChecked():
            self.selection["hcsckl"] = 1
        else:
            self.selection["hcsckl"] = 0
    def hcsckrPush(self,value):
        if self.hcsckr.isChecked():
            self.selection["hcsckr"] = 1
        else:
            self.selection["hcsckr"] = 0

    def confirmPush(self):
        angle_list = []
        for p in self.selection:
            if self.selection[p] == 1:
                angle_list.append(p)

        homedir = os.getcwd()
        # filt = af.Filter(homedir)
        # filt.data_process()
        self.pbar.setValue(25)
        # select = rs.RandomSelect(homedir)
        # select.data_process()
        self.pbar.setValue(50)
        dt = da.DynamicAnalyzer(homedir,angle_list)
        dt.data_process()
        self.pbar.setValue(75)
        c = cl.Classifier(homedir)
        count,rate,total = c.dynamic_classify()
        self.pbar.setValue(100)

        reply = QtGui.QMessageBox.question(self, 'Dynamic Analysis Result',"Total number is %d"%(total)+"\nCorrect number is %d"%(count)+"\nCorrect rate is %f"%(100*rate)+"%", QtGui.QMessageBox.Yes)
        self.pbar.setValue(0)
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = dynamicWindow()
    ex.show()
    app.exec_()