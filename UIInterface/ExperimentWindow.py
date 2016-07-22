#-*-coding:utf-8-*-

import os
import sys
from PyQt4 import QtGui, QtCore
from Window import Window

class ExperimentWindow(Window):
    def __init__(self):
        super(ExperimentWindow, self).__init__()
        from StaticWindow import StaticWindow
        from DynamicWindow import DynamicWindow
        self.staticWindow = StaticWindow()
        self.dynamicWindow = DynamicWindow()
        self.initUI()

    def initUI(self):
        staticAnalysisButton = QtGui.QPushButton("Static Analysis",self)
        staticAnalysisButton.resize(440,100)
        staticAnalysisButton.move(30,100)

        dynamicAnalysisButton = QtGui.QPushButton("Dynamic Analysis",self)
        dynamicAnalysisButton.resize(440,100)
        dynamicAnalysisButton.move(30,300)

        dataInputButton = QtGui.QPushButton("Fusion Analysis",self)
        dataInputButton.resize(440,100)
        dataInputButton.move(30,500)

        self.connect(staticAnalysisButton, QtCore.SIGNAL('clicked()'),self.dataStaticAnalysis)
        self.connect(dynamicAnalysisButton, QtCore.SIGNAL('clicked()'),self.dataDynamicAnalysis)

        self.statusBar().showMessage('Ready')
        self.setWindowTitle("Gait Analysis")

        self.resize(500, 700)
        QtGui.QInputDialog.activateWindow

    def buttonClicked(self):
        sender = self.sender()

    def createDir(self):
        homedir = os.getcwd()
        RawGaitDataFolder = homedir + "\RawGaitDataset"
        fileDir = self.listdir_nohidden(RawGaitDataFolder)
        listdir = []
        for p in fileDir:
            listdir.append(p)
        idnum = len(listdir)+1
        personDir = RawGaitDataFolder + "\Person"+ "%-3d"%(idnum)
        os.mkdir(personDir)
        
    def dataStaticAnalysis(self):
        self.staticWindow.show()
        
    def dataDynamicAnalysis(self):
        self.dynamicWindow.show()

    def fusionAnalysis(self):
        return 0

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = ExperimentWindow()
    ex.show()
    sys.exit(app.exec_())