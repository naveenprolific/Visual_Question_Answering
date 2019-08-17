import warnings,os,sys
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
import spacy, numpy as np
from keras.models import model_from_json, Model
from keras.layers import Dense, Dropout, LSTM, Input, Reshape, concatenate, Activation

from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
from keras.utils import plot_model
#from keras.utils.visualize_util import plot
from sklearn.externals import joblib

from PyQt4 import QtGui
from PyQt4 import QtCore
from sklearn.preprocessing import LabelEncoder

labels = np.load('dataset/encoded_labels.npy')
verbose=1

def get_img_vec(img_path):
    vgg_base_model = VGG16(weights='imagenet')
    
    vgg16_model = Model(vgg_base_model.input, vgg_base_model.layers[-2].output)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    img_vec = np.zeros((1, 4096))
    img_vec[0,:] = vgg16_model.predict(img)[0]
    return img_vec
def get_ques_vec(ques):
    word_embeds = spacy.load('en_vectors_web_lg')
    tokens = word_embeds(ques)
    ques_vec = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
            ques_vec[0,j,:] = tokens[j].vector
    return ques_vec
def model_vqa():
    inp1 = Input((30, 300), dtype=np.float32)
    lstm1 = LSTM(512, return_sequences=True)(inp1)
    lstm2 = LSTM(512, return_sequences=True)(lstm1)
    lstm3 = LSTM(512)(lstm2)

    inp2 = Input(shape=(4096,), dtype=np.float32)
    reshape = Reshape((4096,))(inp2)

    merge = concatenate([lstm3, reshape], axis=1)

    dense1 = Dense(1024, activation='linear')(merge)
    act1 = Activation('tanh')(dense1)
    drop1 = Dropout(0)(act1)
    dense2 = Dense(1024, activation='linear')(drop1)
    act2 = Activation('tanh')(dense2)
    drop2 = Dropout(0)(act2)
    dense3 = Dense(1024, activation='linear')(drop2)
    act3 = Activation('tanh')(dense3)
    drop3 = Dropout(0)(act3)
    dense4 = Dense(1000, activation='linear')(drop3)
    act4 = Activation('softmax')(dense4)
    
    model = Model(inputs=[inp1, inp2], outputs=act4)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.load_weights('dataset/VQA_MODEL_WEIGHTS.hdf5')
    
    return model

vqa_model = model_vqa()
vqa_model.summary()


class VQA_MODEL(QtGui.QWidget):
    
    def __init__(self):
        super(VQA_MODEL, self).__init__()     
        self.initUI()
            
    def initUI(self): 

        self.image_file_name = None
        self.question = None              
        
        self.l1=QtGui.QLabel()
        self.lbl_qstn=QtGui.QLabel()
        self.lbl_output=QtGui.QLabel()
        self.lbl_output.setAlignment(QtCore.Qt.AlignCenter)
      
        self.input_qstn = QtGui.QLineEdit()        
        self.progress = QtGui.QProgressBar(self)
        self.progress.setAlignment(QtCore.Qt.AlignCenter)
        
        font=QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.l1.setFont(font)
        self.l1.setText("<font color='blue'> Choose the image file </font>")
        self.lbl_qstn.setFont(font)
        self.lbl_qstn.setText("<font color='green'> Question </font>")
        self.lbl_output.setFont(font)
        self.lbl_output.setText("<font color='blue'> Answer </font>")
        
        self.te = QtGui.QTextEdit()
        font1 = QtGui.QFont()
        font1.setFamily('Lucida')
        font1.setFixedPitch(True)
        font1.setPointSize(20)
        font1.setBold(True)
        self.te.setFont(font1)
        self.input_qstn.setFont(font1)
        
        self.img_input=QtGui.QLabel()
        self.img_input.resize(self.img_input.sizeHint())  
        self.img_input.setAlignment(QtCore.Qt.AlignCenter)
        
        self.img_output=QtGui.QLabel()
        self.img_output.setAlignment(QtCore.Qt.AlignCenter)
        self.img_output.resize(self.img_output.sizeHint())        
        
        
        self.btn_browse=QtGui.QPushButton("Browse")   
        self.btn_browse.setStyleSheet("background-color: #00bfff")     
        self.btn_browse.clicked.connect(self.Browse)
        self.btn_browse.resize(self.btn_browse.sizeHint())

        self.btn_start=QtGui.QPushButton("PREDICT")   
        self.btn_start.setStyleSheet("background-color: #00ff00")     
        self.btn_start.clicked.connect(self.start_prediction)
        self.btn_start.resize(self.btn_start.sizeHint())  
        
        layout1 = QtGui.QHBoxLayout()
        layout1.addWidget(self.l1)
        layout1.addWidget(self.btn_browse)
        
        layout2 = QtGui.QHBoxLayout()
        layout2.addWidget(self.lbl_qstn)
        layout2.addWidget(self.input_qstn)
          
        vbox_inpt=QtGui.QVBoxLayout()
        vbox_inpt.setMargin(0)
        vbox_inpt.addLayout(layout1)
        vbox_inpt.addLayout(layout2)
        vbox_inpt.addWidget(self.img_input)
        
        vbox_opt=QtGui.QVBoxLayout()
        vbox_opt.setMargin(0)
        vbox_opt.addWidget(self.lbl_output)
        vbox_opt.addWidget(self.progress)
        vbox_opt.addWidget(self.te)
        
        hbox=QtGui.QHBoxLayout()
        hbox.addLayout(vbox_inpt)
        hbox.addLayout(vbox_opt)
        
        vbox_main=QtGui.QVBoxLayout()
        vbox_main.addLayout(hbox)
        vbox_main.addWidget(self.btn_start,5)
        
        self.setLayout(vbox_main)
        self.setGeometry(200, 200, 1200, 700)
        self.setWindowTitle("Visual Question answering Top-5 Predictions")

        self.fname=None
        self.result=None
        
        self.show()     
       
    def Browse(self):

        w = QtGui.QWidget()            
        QtGui.QMessageBox.information(w,"Message", "Please select an image file")          
        
        filePath = QtGui.QFileDialog.getOpenFileName(self, '*.')
        print('filePath',filePath, '\n')
        self.fname=str(filePath)
        self.img_input.setPixmap(QtGui.QPixmap(filePath))
        self.img_input.setScaledContents(True)
        self.image_file_name=self.fname
        
    def start_prediction(self):
        
        self.completed = 0
        self.te.setText('')
            
        self.progress.setValue(15)
        if verbose : print("\n\n\nLoading image features ...")

        img_vector = get_img_vec(self.image_file_name)
        
        self.progress.setValue(40)
        if verbose : print("Loading question features ...")

        
        self.question = self.input_qstn.text()

        ques_vector = get_ques_vec(str(self.question))
    
        self.progress.setValue(100)
        if verbose : print("\n\n\nPredicting result ...")
        preds = vqa_model.predict([ques_vector, img_vector])[0]
        top = np.argsort(preds)[-5:][::-1]
        self.result=[]
        for ind in top:
            print(str(round(preds[ind]*100, 2)) + '%', labels[ind])
            perc = (str(round(preds[ind]*100, 2)) + '%', labels[ind])
            self.result.append(perc)
        self.te.setText('Top 5 predictions : ' + '\n' +  '\n'+str(self.result[0])+ '\n' + str(self.result[1])
        + '\n' + str(self.result[2])+ '\n' + str(self.result[3])+ '\n' + str(self.result[4]))
        
        
def main():
    app = QtGui.QApplication(sys.argv)
    ex = VQA_MODEL()
    app.exec_()

if __name__ == '__main__':
    main()