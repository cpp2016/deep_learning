
from PIL import Image, ImageDraw
import os,cv2,json

#datahome=os.getcwd()
# path of parent dir of current dir 上一级目录所在路径 
datahome=os.path.abspath(os.path.join(os.getcwd(), ".."))

txt=os.path.join(datahome,"train/000000262145.txt")
jpg=os.path.join(datahome,"train/000000262145.jpg")
json_file = os.path.join(datahome,"train/json.json")
#print(txt,jpg)

def parse_json(json_path):
    with open(json_path,"r") as f:
        """
        format of f:
        [
         {"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [x, y], "keypoints": [x1, y1, v1, x2, y2, v2, ..., x18, y18, v18]}]},
         {"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [x, y], "keypoints": [x1, y1, v1, x2, y2, v2, ..., x18, y18, v18]}]},
         ...
         {"filename": "COCO_train2014_000000118171.jpg", "info": [{"pos": [x, y], "keypoints": [x1, y1, v1, x2, y2, v2, ..., x18, y18, v18]}]}
        ]
        """
        jl=json.load(f)
    labeli_dict = list(filter(lambda x:x["filename"]=="000000262145.jpg",jl))[0]
    return labeli_dict

def get_labels(label_file):
    """ cls, box(cx,cy,w,h), cx,cy,w,h>0,<1 """
    ftxt=open(txt,'r')
    lines = ftxt.readlines()
    ftxt.close()
    lb=[]
    for line in lines:
        a = line.strip('\n').split(' ')
        cls,box = int(a[0]), [float(x) for x in a[1:]]
        lb.append((cls,box))
    return lb

def bound(x,boundary=0):
    if(boundary==0):
        return max(x,0)
    else:
        return min(x,boundary-1)
def cxcywh2xyxy(labels,wh):
    """ labels (cls, box(4 01float numbers)); wh, imagesize """
    labels_new = []
    w, h = wh
    for cls, box in labels:
        x1,y1 = int((box[0]-box[2]/2) * w), int( (box[1]-box[3]/2) * h)
        x1,y1 = bound(x1),bound(y1)
        x2,y2 = int((box[0]+box[2]/2) * w), int( (box[1]+box[3]/2) * h)
        x2,y2 = bound(x2,w), bound(y2,h)
        labels_new.append((cls,[x1,y1,x2,y2]))
    return labels_new

def drawbox(labels_new,im):
    dr = ImageDraw.Draw(im)
    for cls, (x1,y1,x2,y2) in labels_new:
        dr.line([(x1,y1),(x1,y2),(x2,y2),(x2,y1),(x1,y1)],width=2,fill='red')
    return im

def drawbox_keypoint(labels_box, labels_keypoint,im):
    dr = ImageDraw.Draw(im)
    for cls, (x1,y1,x2,y2) in labels_box[0:1]:
        dr.line([(x1,y1),(x1,y2),(x2,y2),(x2,y1),(x1,y1)],width=2,fill='red')

    for kpss in labels_keypoint["info"][0:1]:
        kps = kpss["keypoints"]
        for x,y,v in kps:
            #dr.point((x,y),'red')
            box = x-1,y-1,x+1,y+1
            dr.ellipse(box,fill=(0,255,0))
    return im


lbs=get_labels(txt)
im=Image.open(jpg)
wh=im.size
lbs_new = cxcywh2xyxy(lbs,wh)
print(len(lbs_new), lbs_new)
#print(len(lbs), lbs)

ld = parse_json(json_file)

im1 = drawbox_keypoint(lbs_new,ld, im)
im1.show()

