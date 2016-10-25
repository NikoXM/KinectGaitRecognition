#-*-coding:utf-8-*-

import os
import sys
import shutil
from Window import Window
from PyQt4 import QtGui, QtCore

class PracticeWindow(Window):
    def __init__(self):
        super(PracticeWindow, self).__init__()
        from RecognitionWindow import RecognitionWindow
        self.recognitionWindow = RecognitionWindow()
        self.initUI()

    def initUI(self):
        
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

    def createDir(self):
        homedir = os.getcwd()
        RawGaitDataFolder = homedir + "\Dataset\RawGaitDataset"
        fileDir = self.listdirNohidden(RawGaitDataFolder)
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
        exePath = "C:\Users\Niko\Documents\BodyBasics-D2D\Debug\BodyBasics-D2D "
        homedir = os.getcwd()
        outputFilePath = homedir+"\\temp.txt"
        print "outputFilePath: " + outputFilePath
        outputFile = open(outputFilePath,'w')
        outputFile.close()
        #os.system(exePath)
        os.system(exePath+outputFilePath)
        
        personMapFile = open(homedir+"\\PersonMap.txt",'r')
        persons = personMapFile.readlines()
        personMapFile.close()
        isExisted = False

        nameId = -1
        for p in persons:
            p = p.split(" ")
            if p[0] == name:
                nameId = p[1]
                isExisted = True
                break
        if not isExisted:
            print("person not exisist")
            nameId = self.createDir()
            personMapFile = open(homedir+"\\PersonMap.txt",'a')
            personMapFile.write(name+" "+str(nameId)+"\n")
            personMapFile.close()
            personFile = "Person"+"%0*d"%(3,int(nameId))
        else:
            print "nameId: " + nameId
            personFile = "Person"+"%0*d"%(3,int(nameId))
        
        writePath = homedir+"\\Dataset\\RawGaitDataset\\" + personFile
        print "writePath: " + writePath
        fileList = self.listdirNohidden(writePath)
        fileId = len(fileList)+1
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