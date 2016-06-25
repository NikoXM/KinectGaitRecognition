#-*-coding:utf-8-*-

import os
import sys
from PyQt4 import QtGui, QtCore
import dynamicAnalysis as da
import Experiment as exp
import Practice as pra

class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        exit = QtGui.QAction(QtGui.QIcon('icons/exit.png'), 'Exit', self)
        exit.setShortcut('Ctrl+Q')
        exit.setStatusTip('Exit application')
        self.connect(exit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
        self.experiment = exp.Experiment()
        self.practice = pra.Practice()
        self.statusBar()

        menubar = self.menuBar()
        file = menubar.addMenu('&File')
        file.addAction(exit)
        self.initUI()
        
        

    def initUI(self):
        experimentButton = QtGui.QPushButton("Experimental Mode",self)
        experimentButton.resize(940,100)
        experimentButton.move(30,100)

        practiceButton = QtGui.QPushButton("Practical Mode",self)
        practiceButton.resize(940,100)
        practiceButton.move(30,300)

        self.connect(experimentButton, QtCore.SIGNAL('clicked()'),self.experimental)
        self.connect(practiceButton, QtCore.SIGNAL('clicked()'),self.practical)

        self.statusBar().showMessage('Ready')
        self.setWindowTitle("Gait Analysis")

        self.resize(1000, 600)
    def listdir_nohidden(self, path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

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
        
    def experimental(self):
        self.experiment.show()
        
    def practical(self):
        self.practice.show()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())