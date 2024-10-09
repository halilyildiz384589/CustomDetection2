# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:38:45 2023

@author: halil
"""

from ultralytics import YOLO

#Hangi modelle eğitilecek seçiyoruz.
model = YOLO("yolov8m.yaml")

#Use the model
results = model.train(data="data_custom.yaml", epochs=30)

