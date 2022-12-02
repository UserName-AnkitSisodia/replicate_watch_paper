from PIL import Image
import os
import numpy as np
import re
from sklearn import preprocessing
import pandas as pd

path_to_files = os.getcwd()

array_of_images = []
v_wtp = []
v_date = []
v_location = []
v_brand = []
v_circa = []
v_movement = []
v_diameter = []
v_material = []
v_timetrend = []
v_modelname = []
filenames = []
for img in os.listdir(path_to_files):
    if img.endswith("jpg"):
    	name = img
    	wtp_charac = '%'
    	date_charac = '$'
    	location_charac = '#'
    	brand_charac = '^'
    	circa_charac = '@'
    	movement_charac = '!'
    	diameter_charac = '*'
    	material_charac = '<'
    	timetrend_charac = '>'
    	modelname_charac = '|'
    	wtp=name[name.find(wtp_charac)+len(wtp_charac):name.rfind(wtp_charac)]
    	wtp=float(wtp)/4
    	date=name[name.find(date_charac)+len(date_charac):name.rfind(date_charac)]
    	location=name[name.find(location_charac)+len(location_charac):name.rfind(location_charac)]
    	brand=name[name.find(brand_charac)+len(brand_charac):name.rfind(brand_charac)]
    	circa=name[name.find(circa_charac)+len(circa_charac):name.rfind(circa_charac)]
    	movement=name[name.find(movement_charac)+len(movement_charac):name.rfind(movement_charac)]
    	diameter=name[name.find(diameter_charac)+len(diameter_charac):name.rfind(diameter_charac)]
    	diameter=float(diameter)
    	material=name[name.find(material_charac)+len(material_charac):name.rfind(material_charac)]
    	timetrend=name[name.find(timetrend_charac)+len(timetrend_charac):name.rfind(timetrend_charac)]
    	timetrend=float(timetrend)
    	modelname=name[name.find(modelname_charac)+len(modelname_charac):name.rfind(modelname_charac)]
    	single_im = Image.open(img)
    	single_array = np.array(single_im)
    	name = name[name.rfind(modelname_charac)+1:]
    	print(name)
    	if single_array.shape==(128,128,3):
        	array_of_images.append(single_array)
        	v_wtp.append(wtp)
        	v_date.append(date)
        	v_location.append(location)
        	v_brand.append(brand)
        	v_circa.append(circa)
        	v_movement.append(movement)
        	v_diameter.append(diameter)
        	v_material.append(material)
        	v_timetrend.append(timetrend)
        	v_modelname.append(modelname)
        	filenames.append(name)

pre = preprocessing.LabelEncoder()

pre.fit(v_brand)
post_brand=pre.transform(v_brand)
brand_one_hot_encode=pd.get_dummies(post_brand)

pre.fit(v_date)
post_date=pre.transform(v_date)
date_one_hot_encode=pd.get_dummies(post_date)

pre.fit(v_location)
post_location=pre.transform(v_location)
location_one_hot_encode=pd.get_dummies(post_location)

pre.fit(v_circa)
post_circa=pre.transform(v_circa)
circa_one_hot_encode=pd.get_dummies(post_circa)

pre.fit(v_movement)
post_movement=pre.transform(v_movement)
movement_one_hot_encode=pd.get_dummies(post_movement)

pre.fit(v_material)
post_material=pre.transform(v_material)
material_one_hot_encode=pd.get_dummies(post_material)

v_brand = brand_one_hot_encode.to_numpy()
v_date = date_one_hot_encode.to_numpy()
v_location = location_one_hot_encode.to_numpy()
v_circa = circa_one_hot_encode.to_numpy()
v_movement = movement_one_hot_encode.to_numpy()
v_material = material_one_hot_encode.to_numpy()

v_brand = np.delete(v_brand,np.s_[2],axis=1)
v_location = np.delete(v_location,np.s_[3],axis=1)
v_circa = np.delete(v_circa,np.s_[7],axis=1)
v_movement = np.delete(v_movement,np.s_[2],axis=1)
v_material = np.delete(v_material,np.s_[2],axis=1)

np.savez("christies.npz",watches=array_of_images,wtp=v_wtp,diameter=v_diameter,timetrend=v_timetrend,modelname=v_modelname,date=v_date,location=v_location,brand=v_brand,circa=v_circa,movement=v_movement,material=v_material,filenames=filenames)
