import os,sys,re,cPickle
import xml.etree.ElementTree as ET
import numpy as np                 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

             
def xml_to_id(xml_str):
    mat = re.match(r".*/([0-9]{4}_)?(?P<id>[0-9A-Za-z_]+)\.[A-Za-z]+",xml_str)
    if mat == None:
        raise ValueError("xml_file {0:s} does not match regex".format(xml_str))
    return mat.groupdict()["id"]

def parse_rec_xml(filename):           
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

def print_and_save(aps,ovthresh,classes,output_dir):
    print(aps.shape)
    print('~~~~~~~~')
    print('Results:')
    sys.stdout.write('{0:>15}:'.format("class"))
    for thsh in ovthresh:
        sys.stdout.write("\t@{0:>10.3f}".format(thsh))
    sys.stdout.write("\n")
    print('~~~~~~~~')
    for idx,ap in enumerate(aps):
        sys.stdout.write('{0:>15}:\t'.format(classes[idx]))
        for kdx in range(len(ovthresh)):
            #sys.stdout.write('{0:.5f} @ {1:.2f}\t'.format(ap[kdx],ovthresh[kdx]))
            sys.stdout.write('{0:>10.5f}\t'.format(ap[kdx],ovthresh[kdx]))
        with open(os.path.join(output_dir, classes[idx] + '_pr.pkl'), 'w') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        sys.stdout.write('\n')
    print('~~~~~~~~')
    print("Overall mAP")
    for kdx in range(len(ovthresh)):
        print('Mean AP = {:.4f} @ {:.2f}'.format(np.nanmean(aps[:,kdx]),ovthresh[kdx]))

def voc_ap(rec, prec, classname, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def voc_eval(pred_dets_file,
             xml_files,
             classname,
             cachedir, 
             ovthresh=0.5, 
             use_07_metric=False): 
    """rec, prec, ap = voc_eval(detpath, 
        annopath,
        classname, 
        [ovthresh],
        [use_07_metric]) 
        Top level function that does the PASCAL VOC evaluation.
        detpath: Path to detections
        detpath.format(classname) should produce the detection results file. 
        annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file. 
        classname: Category name (duh) 
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name 
 
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i,xml_file in enumerate(xml_files):
            # try:
            #     recs[xml_file] = parse_rec(xml_file)
            # except:
            #     print(xml_file)
            recs[xml_to_id(xml_file)] = parse_rec_xml(xml_file)
            if i % 100 == 0: 
                print 'Reading annotation for {:d}/{:d}'.format( 
                    i + 1, len(xml_files))
        # save 
        print('Saving cached annotations to {:s}').format(cachefile)
        cPickle.dump(recs,open(cachefile, 'w'))
    else:
        # load 
        recs = cPickle.load(open(cachefile, 'r'))
    
    # extract gt objects for this class
    class_recs = {}
    npos = 0 
    for xml_file in xml_files:#imagenames: 
        imagename = xml_to_id(xml_file)
        R = [obj for obj in recs[imagename] if obj['name'] == classname] 
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R) 
        npos = npos + sum(~difficult) + sum(difficult)
        class_recs[imagename] = {'bbox': bbox, 
                             'difficult': difficult, 
                             'det': det} 
    # read dets
    with open(pred_dets_file, 'r') as f:
        lines = f.readlines()
    
    splitlines = [x.strip().split(' ') for x in lines] 
    image_ids = [x[0] for x in splitlines] 
    if "aeroplane" == classname: print(image_ids[0:10])
    confidence = np.array([round(float(x[1]),5) for x in splitlines])
    if "aeroplane" == classname: print(confidence[0:10])
    #BB = np.array([[round(float(z),1) for z in x[2:]] for x in splitlines])
    BB = np.array([[round(float(z),5) for z in x[2:]] for x in splitlines])
    if "aeroplane" == classname: print(BB[0:10])

    # sort by confidence 
    sorted_ind = np.argsort(-confidence) 
    sorted_scores = np.sort(-confidence) 
    if "aeroplane" == classname: print(np.array(sorted_scores).shape)
    if len(BB) == 0: 
        return 0,0,0 
    BB = BB[sorted_ind, :] 
    image_ids = [image_ids[x] for x in sorted_ind] 

    #ovthresh = [0.5,0.75,0.95] 
    ovthresh = [0.25,0.5,0.75,0.95] 
    nd = len(image_ids)
    tp = np.zeros((nd,len(ovthresh)))
    fp = np.zeros((nd,len(ovthresh)))
    for d in range(nd):
        R = class_recs[image_ids[d]] 
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float) 
        if BBGT.size > 0:
            # compute overlaps 
            # intersection 
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih 
 
            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + 
             (BBGT[:, 2] - BBGT[:, 0] + 1.) *
             (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps) 
            jmax = np.argmax(overlaps) 
        # if(sorted_scores[d] >= -0.5):
        #     continue 
        # if ovmax > 0.5:
        #     print(ovmax)
        #print(sorted_scores[d],sorted_scores[d] < -0.0) 
        inside_any = False 
        for idx in range(len(ovthresh)): 
            if ovmax > ovthresh[idx]:
                #if not R['difficult'][jmax]: 
                if True:
                    if not R['det'][jmax]: 
                        inside_any = True 
                        tp[d,idx] = 1. 
                        #print("tp") 
                    else:
                        fp[d,idx] = 1. 
                        #print("fp") 
                else:
                    fp[d,idx] = 1. 
                    #print("fp") 
 
        if inside_any is True: 
            R['det'][jmax] = 1 
    
 
    rec = np.zeros((len(fp),len(ovthresh)))
    prec = np.zeros((len(fp),len(ovthresh))) 
    ap = np.zeros(len(ovthresh)) 
    for idx in range(len(ovthresh)): 
        # compute precision recall 
        _fp = np.cumsum(fp[:,idx]) 
        _tp = np.cumsum(tp[:,idx]) 
        rec[:,idx] = _tp / float(npos) 
        # avoid divide by zero in case the first detection matches a difficult 
        # ground truth 
        prec[:,idx] = _tp / np.maximum(_tp + _fp, np.finfo(np.float64).eps)
        #ap = voc_ap(rec, prec, use_07_metric) 
        ap[idx] = voc_ap(rec[:,idx], prec[:,idx], classname, False)
 
    print("tp",np.sum(tp))
    print("fp",np.sum(fp))
    #print(ap)
    #print(fp,tp,rec,prec,ap,npos) 
    return rec, prec, ap, ovthresh


if __name__ == "__main__":


    img_files = open("voc_2007_test.txt","r").read().split("\n")[:-1]
    xml_files = open("voc_2007_test.txt","r").read()\
                .replace("JPEGImages","Annotations")\
                .replace("jpg","xml")\
                .replace("jpeg","xml")\
                .replace("png","xml").split("\n")[:-1]
    classes = open("voc.names","r").read().split("\n")[:-1]
    voc_classes = open("voc.names","r").read().split("\n")[:-1]
    cachefile = "anno_cache.pkl"
    output_dir = "score"

    aps = []
    for i, cls in enumerate(classes):
        # if cls not in voc_classes:
        #     continue
        # filename = "results/comp4_det_test_{0:s}.txt".format(cls) 
        filename = "results_10000/comp4_det_test_{0:s}.txt".format(cls) 
        # filename = "gt_files/{0:s}_gt.txt".format(cls) 
        rec, prec, ap, ovthresh = voc_eval(
            filename, xml_files, cls, cachefile, ovthresh=0.5,use_07_metric=False)
        aps += [ap]
    aps = np.array(aps) 
    print_and_save(aps,ovthresh,classes,output_dir)
