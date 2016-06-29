#-*-coding:utf-8-*-

import os
import sys
import shutil
from Window import Window

class PracticeWindow(Window):
    def __init__(self):
        super(PracticeWindow, self).__init__()
        self.initUI()

    def initUI(self):
        from PyQt4 import QtGui, QtCore
        dataInputButton = QtGui.QPushButton("Data Input",self)
        dataInputButton.resize(440,100)
        dataInputButton.move(30,100)
        
        gatiRecognitionButton = QtGui.QPushButton("Gait Recognition",self)
        gatiRecognitionButton.resize(440,100)
        gatiRecognitionButton.move(30,300)

        showResultButton = QtGui.QPushButton("Show Result",self)
        showResultButton.resize(440,100)
        showResultButton.move(30,500)

        self.connect(dataInputButton, QtCore.SIGNAL('clicked()'),self.dataInput)
        self.connect(gatiRecognitionButton, QtCore.SIGNAL('clicked()'),self.gaitRecognitionPush)
        self.connect(showResultButton, QtCore.SIGNAL('clicked()'),self.showResultPush)

        self.statusBar().showMessage('Ready')
        self.setWindowTitle("Gait Analysis")

        self.resize(500, 800)

        from RecognitionWindow import RecognitionWindow
        self.recognitionWindow = RecognitionWindow()

    def createDir(self):
        homedir = os.getcwd()
        RawGaitDataFolder = homedir + "\RawGaitDataset"
        fileDir = self.listdir_nohidden(RawGaitDataFolder)
        listdir = []
        for p in fileDir:
            listdir.append(p)
        idnum = len(listdir)+1
        personDir = RawGaitDataFolder + "\Person"+ "%0*d"%(3,idnum)
        os.mkdir(personDir)
        return idnum

    def dataInput(self):
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + ' is working')

        name = self.showDialog()
        isok = True
        while name == "":
            reply = QtGui.QMessageBox.question(self, 'Errors',"Empty Name if rename?!", QtGui.QMessageBox.Yes,QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                name = self.showDialog()
            else:
                isok = False
                break
        if not isok:
            return
        exePath = "C:\Users\Niko\Documents\BodyBasics-D2D\Debug\BodyBasics-D2D"
        homedir = os.getcwd()
        outputFilePath = homedir+"\\temp.txt"
        outputFile = open(outputFilePath,'w')
        outputFile.close()
        os.system(exePath+outputFilePath)
        
        personMapFile = open(homedir+"\\PersonMap.txt",'r')
        persons = personMapFile.readlines()
        personMapFile.close()
        isExisted = False
        for p in persons:
            p = p.split(" ")
            if p[0] == name:
                nameId = p[1]
                isExisted = True
        if not isExisted:
            nameId = self.createDir()
            personMapFile = open(homedir+"\\PersonMap.txt",'a')
            personMapFile.write(name+" "+str(nameId)+"\n")
            personMapFile.close()
            personFile = "Person"+"%0*d"%(3,int(nameId))
        else:
            personFile = "Person"+"%0*d"%(3,int(p[1]))
        
        writePath = homedir+"\\RawGaitDataset\\"+personFile
        files = self.listdir_nohidden(writePath)
        file_list = []
        for f in files:
            file_list.append(f)
        fileId = len(file_list)+1

        filePath = writePath +"\\" +str(fileId)+".txt"
        shutil.move(outputFilePath,filePath)
        
    def showDialog(self):
        text, ok = QtGui.QInputDialog.getText(self, 'Input Dialog','Enter your name:')
        if ok:
            return str(text)
        else:
            return ""
        
    def gaitRecognitionPush(self):
        self.recognitionWindow.show()

    def showResultPush(self):
        return 0

if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    ex = PracticeWindow()
    ex.show()
    sys.exit(app.exec_())