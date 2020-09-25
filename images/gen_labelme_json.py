import os
from glob import glob
import math
import json
import base64
from PIL import Image

dt_types=['train','test']

for dt_type in dt_types:
    
    gt_files = glob(os.path.join(dt_type,"*.gt"))
    for gt_file in gt_files:
        json_file = gt_file.replace(".gt",".json")
        img_file = os.path.basename(gt_file).replace(".gt",".JPG")
        with open(dt_type+"/"+img_file,'rb') as imgf:
            data = imgf.read()
            imgbase64 = bytes.decode(base64.b64encode(data))
            
        img = Image.open(dt_type+"/"+img_file)
        img_width = img.width
        img_height = img.height
        labelme_obj = {"version": "4.5.6",
                                "flags": {},
                                "imagePath":img_file,
                                "imageData":imgbase64,
                                "imageHeight":img_height,
                                "imageWidth":img_width}
                                
        with open(gt_file,'r') as gf:
            shapes = []
            for line in gf.readlines():
                
                items = line.strip().split(" ")
                id = items[0]
                diff = items[1]
                if diff=='1':
                    continue
                x = int(items[2])
                y = int(items[3])
                width = int(items[4])
                height = int(items[5])
                angle = float(items[6])
                shape = {"label": "txt","group_id": None,"shape_type": "polygon","flags": {}}
                x1 = x
                y1 = y
                x2 = x+width
                y2 = y
                x3 = x2
                y3 = y+height
                x4 = x1
                y4 = y3
                cx = x+width//2
                cy = y+height//2
                
                x1 = x1-cx
                y1 = y1-cy
                
                x2 = x2-cx
                y2 = y2-cy
                
                x3 = x2
                y3 = y3-cy
                
                x4 = x1
                y4 = y3
                
                
                r_x1 = x1*math.cos(angle)-y1*math.sin(angle)
                r_y1 = y1*math.cos(angle)+x1*math.sin(angle)
                
                r_x2 = x2*math.cos(angle)-y2*math.sin(angle)
                r_y2 = y2*math.cos(angle)+x2*math.sin(angle)
                
                r_x3 = x3*math.cos(angle)-y3*math.sin(angle)
                r_y3 = y3*math.cos(angle)+x3*math.sin(angle)
                
                r_x4 = x4*math.cos(angle)-y4*math.sin(angle)
                r_y4 = y4*math.cos(angle)+x4*math.sin(angle)
                
                r_x1 = r_x1+cx
                r_y1 = r_y1+cy
                
                r_x2 = r_x2+cx
                r_y2 = r_y2+cy
                
                r_x3 = r_x3+cx
                r_y3 = r_y3+cy
                
                r_x4 = r_x4+cx
                r_y4 = r_y4+cy
                
                
                
                shape["points"] = [[r_x1,r_y1],[r_x2,r_y2],[r_x3,r_y3],[r_x4,r_y4]]
                
                shapes.append(shape)
        
        labelme_obj["shapes"] = shapes   
        
        with open(json_file,'w',encoding='utf-8') as jf:
            jf.write(json.dumps(labelme_obj))
            
                