import os,sys,re,cPickle,json
import numpy as np                 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

             
def txt_to_id(txt_str):
    mat = re.match(r".*/([0-9]{4}_)?(?P<id>[0-9A-Za-z_]+)\.[A-Za-z]+",txt_str)
    if mat == None:
        raise ValueError("txt_file {0:s} does not match regex".format(txt_str))
    return mat.groupdict()["id"]

def parse_rec_txt(filename,classes):
    """ Parse a COCO txt file """                  
    try:
        annos = open(filename,"r").read().split("\n")[:-1]
    except:
        print("filename: {} not found".format(filename))
        return None
    objects = []                   
    for anno in annos:
        anno = anno.split(" ")
        obj = {}
        obj["name"] = classes[int(float(anno[0]))]
        obj["bbox"] =  [float(anno[i+1]) for i in range(4)]
        objects.append(obj)
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



def coco_eval(pred_dets_file,
              txt_files,
              classes,
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
        annopath.format(imagename) should be the txt annotations file. 
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
        for i,txt_file in enumerate(txt_files):
            # try:
            #     recs[txt_file] = parse_rec(txt_file)
            # except:
            #     print(txt_file)
            recs[txt_to_id(txt_file)] = parse_rec_txt(txt_file,classes)
            if i % 100 == 0: 
                print 'Reading annotation for {:d}/{:d}'.format( 
                    i + 1, len(txt_files))
        # save 
        print('Saving cached annotations to {:s}').format(cachefile)
        cPickle.dump(recs,open(cachefile, 'w'))
    else:
        # load 
        recs = cPickle.load(open(cachefile, 'r'))
    
    # extract gt objects for this class
    class_recs = dict.fromkeys(classes,{})
    npos = dict.fromkeys(classes,0)
    for txt_file in txt_files:#imagenames: 
        imagename = txt_to_id(txt_file)
        for cls in classes:
            if not recs[imagename]:
                continue
            R = [obj for obj in recs[imagename] if obj['name'] == cls] 
            bbox = np.array([x['bbox'] for x in R])
            det = [False] * len(R) 
            npos[cls] += len(bbox)
            class_recs[cls][imagename] = {'bbox': bbox, 
                                          'det': det} 
    # read dets
    with open(pred_dets_file, 'r') as f:
        preds = json.load(f)
    image_ids = dict.fromkeys(classes,[])
    pred_class = dict.fromkeys(classes,[])
    confidence = dict.fromkeys(classes,[])
    BB = dict.fromkeys(classes,[])
    for pred_info in preds:
        if pred_info["category_id"] > 80:
            continue
        cls = classes[pred_info["category_id"]-1]
        image_ids[cls] += [pred_info["image_id"]]
        confidence[cls] += [pred_info["score"]]
        BB[cls] += [pred_info["bbox"]]
            
    recs = dict.fromkeys(classes,None)
    precs = dict.fromkeys(classes,None)
    aps = dict.fromkeys(classes,None)
    ovthresh = dict.fromkeys(classes,None)
    for cls in classes:
        recs[cls],precs[cls],aps[cls],ovthresh[cls] = get_ap_cls(np.array(image_ids[cls]),np.array(confidence[cls]),np.array(BB[cls]),class_recs[cls],npos[cls],cls)
    return recs,precs,aps,ovthresh[cls]

def get_ap_cls(image_ids,confidence,BB,class_recs,npos,classname):

    # sort by confidence 
    sorted_ind = np.argsort(-confidence) 
    sorted_scores = np.sort(-confidence) 
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
        try:
            R = class_recs[image_ids[d]]
        except:
            continue
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


def mangle_img_f(f):
    return f.replace("images","labels")\
                .replace("jpg","txt")\
                .replace("jpeg","txt")\
                .replace("png","txt")

if __name__ == "__main__":


    img_files = open("coco/5k.txt","r").read().split("\n")[:-1]
    txt_files = open("coco/5k.txt","r").read()\
                .replace("images","labels")\
                .replace("jpg","txt")\
                .replace("jpeg","txt")\
                .replace("png","txt").split("\n")[:-1]
    # not all txt_files exist
    classes = open("coco.names","r").read().split("\n")[:-1]
    cachefile = "coco_anno_cache.pkl"
    output_dir = "coco_score"
    filename = "results_coco_10k/coco_results.json"
    rec, prec, aps, ovthresh = coco_eval(
        filename, txt_files, classes, cachefile, ovthresh=0.5,use_07_metric=False)

    aps = np.array(aps) 
    print_and_save(aps,ovthresh,classes,output_dir)
