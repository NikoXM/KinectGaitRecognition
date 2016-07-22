import os
from PyQt4 import QtGui, QtCore
class Window(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        #MARK: Property
        
        #MARK: Action
        #exit
        exit = QtGui.QAction(QtGui.QIcon('icons/exit.png'), 'Exit', self)
        exit.setShortcut('Ctrl+Q')
        exit.setStatusTip('Exit application')
        self.connect(exit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
        
        #MARK: Menu
        #File
        menubar = self.menuBar()
        file = menubar.addMenu('&File')
        file.addAction(exit)
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Window")
        self.resize(1000, 600)
    
    def listdirNohidden(self, path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f
        

if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())