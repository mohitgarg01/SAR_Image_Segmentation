# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 22:31:32 2018

@author: win 8.1
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tkinter.simpledialog

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

def getorigin(eventorigin):
      x = eventorigin.x
      y = eventorigin.y
      print(x,y)
      print('Start Mouse Position: ' + str(x) + ', ' + str(y))
      s_box = x, y
      boxes.append(s_box)

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items==[]

    def enque(self,item):
        self.items.insert(0,item)

    def deque(self):
        return self.items.pop()

    def qsize(self):
        return len(self.items)
    
    def isInside(self, item):
        return (item in self.items)
        
    
def regiongrow(image,epsilon,start_point):

    Q = Queue()
    s = []
    
    x = start_point[0]
    y = start_point[1]
    
    Q.enque((x,y))
    
    while not Q.isEmpty():

        t = Q.deque()
        x = t[0]
        y = t[1]
        
        if x < image.size[0]-1 and \
           abs(  image.getpixel( (x + 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon :

            if not Q.isInside( (x + 1 , y) ) and not (x + 1 , y) in s:
                Q.enque( (x + 1 , y) )

                
        if x > 0 and \
           abs(  image.getpixel( (x - 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon:

            if not Q.isInside( (x - 1 , y) ) and not (x - 1 , y) in s:
                Q.enque( (x - 1 , y) )

                     
        if y < (image.size[1] - 1) and \
           abs(  image.getpixel( (x , y + 1) ) - image.getpixel( (x , y) )  ) <= epsilon:

            if not Q.isInside( (x, y + 1) ) and not (x , y + 1) in s:
                Q.enque( (x , y + 1) )

                    
        if y > 0 and \
           abs(  image.getpixel( (x , y - 1) ) - image.getpixel( (x , y) )  ) <= epsilon:

            if not Q.isInside( (x , y - 1) ) and not (x , y - 1) in s:
                Q.enque( (x , y - 1) )

        if t not in s:
            s.append( t )

            
    return s

    
    
    

def getSegmentedImage(File, boxes):
    img = Image.open(File)
    img = img.resize((256, 256), Image.ANTIALIAS)
    resized = img.convert('L')
    final = np.zeros((256,256),np.uint8)
    imag = np.array(list(resized.getdata(band=0)), float)
    imag.shape = (resized.size[1], resized.size[0])
    plt.imshow(imag)
    c = 0
    diff = int(255 / len(boxes))
    for i in range ( 256 ):
        for j in range ( 256 ):
            final[i][j] = 0
    for box in boxes:
        print("Pixel", img.getpixel((20,20)))
        color = 255 - c*diff
        print("color is ", color)
        c = c+1
        s = regiongrow(resized, 5, box)
        for i in s:
            final[i[0]][i[1]] = color
    
    
    return final

if __name__ == '__main__':

    boxes = []
    
    root = Tk()
    #setting up a tkinter canvas
    w = Canvas(root, width=256, height=256)
    w.pack()
    
    File = askopenfilename(parent=root, initialdir="./",title='Select an image')
    original = Image.open(File)
    original = original.resize((256,256)) #resize image
    img = ImageTk.PhotoImage(original)
    w.create_image(0, 0, image=img, anchor="nw")

    root.bind("<Button 1>",getorigin)
    root.mainloop()
    print(boxes)
    
    fin = getSegmentedImage(File, boxes)
    img = Image.fromarray(fin)
    img.show()
    clust = 3
    mask = fin.astype(bool)
    img = fin.astype(float)
    graph = image.img_to_graph(img, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())
    labels = spectral_clustering(graph, n_clusters=clust, eigen_solver='arpack')
    label_im = np.full(mask.shape, -1.)
    label_im[mask] = labels

    plt.matshow(label_im)
    for i in range(256):
        for j in range(256):
            label_im[i][j] = int((label_im[i][j]+1)*(255/(clust+1)))
    label_img = Image.fromarray(label_im)
    label_img.show()
    