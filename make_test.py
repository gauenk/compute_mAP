import os,sys,re
import xml.etree.ElementTree as ET
import numpy as np                 


def xml_to_id(xml_str):
    mat = re.match(r".*/([0-9]{4}_)?(?P<id>[0-9A-Za-z_]+)\.[A-Za-z]+",xml_str)
    if mat == None:
        raise ValueError("xml_file {0:s} does not match regex".format(xml_str))
    return mat.groupdict()["id"]


def parse_rec(filename):           
    """ Parse a PASCAL VOC xml file """                  
    tree = ET.parse(filename)      
    objects = []                   
    for obj in tree.findall('object'):                   
        obj_struct = {}            
        obj_struct['name'] = obj.find('name').text       
        obj_struct['pose'] = obj.find('pose').text       
        obj_struct['truncated'] = int(obj.find('truncated').text)              
        obj_struct['difficult'] = int(obj.find('difficult').text)              
        bbox = obj.find('bndbox')  
        obj_struct['bbox'] = [int(bbox.find('xmin').text),                     
        int(bbox.find('ymin').text),                     
        int(bbox.find('xmax').text),                     
        int(bbox.find('ymax').text)]                     
        objects.append(obj_struct)
    return objects

def open_cls_files(classes,gt_dir):
    fps = {}
    for cls in classes:
        fps[cls] = open(os.path.join(gt_dir,"{}_gt.txt".format(cls)),"w")
    return fps

def close_fps(fps):
    for fp in fps.values():
        fp.close()

def save_recs(fps,recs,classes,f_id):
    for rec in recs:
        fps[rec["name"]].write("{0:s} {1:.4f} {2[0]:.4f} {2[1]:.4f} {2[2]:.4f} {2[3]:.4f}\n"\
            .format(f_id,1.0,rec["bbox"]))

if __name__ == "__main__":

    print("hi taylor")
    xml_files = open("voc_2007_test.txt","r").read()\
                .replace("JPEGImages","Annotations")\
                .replace("jpg","xml")\
                .replace("jpeg","xml")\
                .replace("png","xml").split("\n")[:-1]
    classes = open("voc.names","r").read().split("\n")[:-1]
    gt_dir = "gt_files"
    fps = open_cls_files(classes,gt_dir)
    
    for xml_file in xml_files:
        recs = parse_rec(xml_file)
        save_recs(fps,recs,classes,xml_to_id(xml_file))

    close_fps(fps)
    
