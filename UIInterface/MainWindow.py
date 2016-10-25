#-*-coding:utf-8-*-
import os
import sys
homdir = sys.path[0]
i = homdir.rfind('\\')
homdir = homdir[0:i]
sys.path.append(homdir)
sys.path.append(homdir+"\\ProcessLogic")
print homdir

from PyQt4 import QtGui, QtCore
from Window import Window

class MainWindow(Window):
    def __init__(self):
        super(MainWindow,self).__init__() 
        from ExperimentWindow import ExperimentWindow
        from PracticeWindow import PracticeWindow
        self.experimentWindow = ExperimentWindow()
        self.practiceWindow = PracticeWindow()
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

    def createDir(self):
        RawGaitDataFolder = homedir + "\RawGaitDataset"
        fileDir = self.listdir_nohidden(RawGaitDataFolder)
        listdir = []
        for p in fileDir:
            listdir.append(p)
        idnum = len(listdir)+1
        personDir = RawGaitDataFolder + "\Person"+ "%-3d"%(idnum)
        os.mkdir(personDir)
        
    def experimental(self):
        self.experimentWindow.show()
        
    def practical(self):
        self.practiceWindow.show()

if __name__ == "__main__":
    import sys
    homdir = sys.path[0]
    i = homdir.rfind('\\')
    homdir = homdir[0:i]
    sys.path.append(homdir)
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())