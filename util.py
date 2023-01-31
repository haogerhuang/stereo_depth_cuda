import numpy as np
import cv2
import os

class Dataset:
    def __init__(self, name):
        self.img0 = cv2.imread(os.path.join(name, 'im0.png'))
        self.img1 = cv2.imread(os.path.join(name, 'im1.png'))
        self.calib = Calib(os.path.join(name, 'calib.txt'))
        self.name = name

    @property
    def gray_img0(self):
        return cv2.cvtColor(self.img0, cv2.COLOR_BGR2GRAY)

    @property
    def gray_img1(self):
        return cv2.cvtColor(self.img0, cv2.COLOR_BGR2GRAY)

class Calib:
    def __init__(self, name):
        self.config = {}
        self.read_calib(name)

    def read_calib(self, name):
        f = open(name, 'r')
        data = f.readlines()
        for line in data:
            key, value = line.rstrip().split('=')
            self.config[key] = value
            
    @classmethod
    def parseStr2Array(cls, str_array):
        res = []
        str_array = str_array[1:-1]
        rows = str_array.split(';')
        for row in rows:
            res.append(np.fromstring(row, dtype=float, sep=' '))
        return np.array(res)        
            
    @property
    def cam0(self):
        return self.parseStr2Array(self.config['cam0'])
        
    @property
    def cam1(self):
        return self.parseStr2Array(self.config['cam1'])

    @property
    def baseline(self):
        return float(self.config['baseline'])

    @property
    def ndisp(self):
        return int(self.config['ndisp'])

    @property
    def vmin(self):
        return int(self.config['vmin'])

    @property
    def vmax(self):
        return int(self.config['vmax'])
    
def toHomogeneous(data):
    N = data.shape[0]
    homo_data = np.ones([N,3])
    homo_data[:,0:2] = data
    return homo_data

