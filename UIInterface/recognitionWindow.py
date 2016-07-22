import os
import shutil
from PyQt4 import QtGui, QtCore
from StaticAnalyzer import StaticAnalyzer
from DynamicAnalyzer import DynamicAnalyzer
from Classifier import Classifier
from Window import Window

limbDescriptors  = ['neck','rshoulder','lshoulder','rarm','larm','rfarm','lfarm',
					'rhand','lhand','uspine','lspine','rhip','lhip','rthigh','lthigh',
					'rcalf','lcalf','rfoot','lfoot','height']
angleDescriptors = ['srkrar','srklal','slkrar','slklal','hrklal','hlkrar','krhlal','klhrar','arhlkl','alhrkr']

class RecognitionWindow(Window):
    def __init__(self):
        super(RecognitionWindow, self).__init__()
        self.initUI()
        
    def initUI(self):

        self.staticRecognition = QtGui.QCheckBox("static recognition", self)
        self.staticRecognition.move(30,150)

        self.dynamicRecognition = QtGui.QCheckBox('dynamic recognition', self)
        self.dynamicRecognition.move(30,200)

        self.fusionRecognition = QtGui.QCheckBox('fusion recognition', self)
        self.fusionRecognition.move(30,250)

        self.confirmButton = QtGui.QPushButton("Confirm",self)
        self.confirmButton.resize(200,80)
        self.confirmButton.move(30,600)

        self.cancelButton = QtGui.QPushButton("cancel",self)
        self.cancelButton.resize(200,80)
        self.cancelButton.move(300,600)

        self.connect(self.staticRecognition, QtCore.SIGNAL('stateChanged(int)'),self.staticPush)
        self.connect(self.dynamicRecognition, QtCore.SIGNAL('stateChanged(int)'),self.dynamicPush)
        self.connect(self.fusionRecognition, QtCore.SIGNAL('stateChanged(int)'),self.fusionPush)

        self.connect(self.confirmButton,QtCore.SIGNAL('clicked()'),self.confirmPush)
        self.connect(self.cancelButton,QtCore.SIGNAL('clicked()'), QtCore.SLOT('close()'))

        self.statusBar().showMessage('Ready')
        self.setWindowTitle("Gait Recognition")

        self.pbar = QtGui.QProgressBar(self)
        self.pbar.setGeometry(30, 60, 500, 25)

        self.resize(500, 800)
        self.mode = []

    def findName(self,i):
        homdir = os.getcwd()
        personMapPath = homdir+"\personMap.txt"
        personMapFile = open(personMapPath)
        personMap = personMapFile.readlines()
        personMapFile.close()
        for pm in personMap:
            pm = pm.split(' ')
            if str(pm[1].replace('\n','')) == str(i):
                return pm[0]
        return str(i)

    def staticPush(self):
        if self.staticRecognition.isChecked():
            self.dynamicRecognition.setCheckState(0)
            self.fusionRecognition.setCheckState(0)

    def dynamicPush(self):
        if self.dynamicRecognition.isChecked():
            self.staticRecognition.setCheckState(0)
            self.fusionRecognition.setCheckState(0)

    def fusionPush(self):
        if self.fusionRecognition.isChecked():
            self.dynamicRecognition.setCheckState(0)
            self.staticRecognition.setCheckState(0)

    def confirmPush(self):
        checked = self.staticRecognition.isChecked() or self.dynamicRecognition.isChecked() or self.fusionRecognition.isChecked()
        if not checked:
            reply = QtGui.QMessageBox.question(self, 'Analysis Result',"Select One", QtGui.QMessageBox.Yes)
            return

        homdir = os.getcwd()
        trainGaitPath = homdir+"\\TrainDataset\\TrainGaitDataset"
        if (os.path.exists(trainGaitPath)):
            shutil.rmtree(trainGaitPath)
            os.mkdir(trainGaitPath)
        else:
            os.mkdir(trainGaitPath)
        filterFilePath = homdir+"\\FilteredGaitDataset"
        files = os.listdir(filterFilePath)
        for f in files:
            fpath = filterFilePath + "\\"+f
            dstGaitPath = trainGaitPath+"\\"+f
            shutil.copytree(fpath,dstGaitPath)

        testGaitPath = homdir+"\\TestDataset\\TestGaitDataset"
        if (os.path.exists(testGaitPath)):
            shutil.rmtree(testGaitPath)
            os.mkdir(testGaitPath)
        else:
            os.mkdir(testGaitPath)
        
        exePath = "C:\Users\Niko\Documents\BodyBasics-D2D\Debug\BodyBasics-D2D "
        homdir = os.getcwd()
        outputFilePath = homdir+"\\test.txt"
        # outputFile = open(outputFilePath,'w')
        # outputFile.close()
        # os.system(exePath+outputFilePath)

        dstOutputPersonPath = homdir+"\\TestDataset\\TestGaitDataset\\Person001"
        os.mkdir(dstOutputPersonPath)
        dstOutputPath = dstOutputPersonPath+"\\1.txt"
        shutil.copy(outputFilePath,dstOutputPath)

        self.pbar.setValue(50)
        if self.staticRecognition.isChecked():
            self.pbar.setValue(75)
            st = StaticAnalyzer(homdir,limbDescriptors)
            st.data_process()
            c = cl.Classifier(homdir)
            count,rate,total,result = c.staticClassify()
            self.pbar.setValue(100)
            name = self.findName(result[0])
            reply = QtGui.QMessageBox.question(self, 'Static Analysis Result',"This is "+name, QtGui.QMessageBox.Yes)
            
        elif self.dynamicRecognition.isChecked():
            self.pbar.setValue(75)
            dy = DynamicAnalyzer(homdir,angleDescriptors)
            dy.data_process()
            c = Classifier(homdir)
            count,rate,total = c.dynamicClassify()
            self.pbar.setValue(100)
            reply = QtGui.QMessageBox.question(self, 'Dynamic Analysis Result',"Total number is %d"%(total)+"\nCorrect number is %d"%(count)+"\nCorrect rate is %f"%(100*rate)+"%", QtGui.QMessageBox.Yes)
        else:
            self.pbar.setValue(75)
            dy = DynamicAnalyzer(homdir,angleDescriptors)
            dy.dataProcess()
            st = StaticAnalyzer(homdir,limbDescriptors)
            st.dataProcess()
            c = Classifier(homdir)
            count,rate,total = c.fusionClassify()
            self.pbar.setValue(100)
            reply = QtGui.QMessageBox.question(self, 'Fusion Analysis Result',"Total number is %d"%(total)+"\nCorrect number is %d"%(count)+"\nCorrect rate is %f"%(100*rate)+"%", QtGui.QMessageBox.Yes)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = RecognitionWindow()
    ex.show()
    sys.exit(app.exec_())