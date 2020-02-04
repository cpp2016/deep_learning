
from PIL import Image, ImageDraw
import os,cv2

#datahome=os.getcwd()
# path of parent dir of current dir 上一级目录所在路径 
datahome=os.path.abspath(os.path.join(os.getcwd(), ".."))

txt=os.path.join(datahome,"train/000000000077.txt")
jpg=os.path.join(datahome,"train/000000000077.jpg")
#print(txt,jpg)

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

lbs=get_labels(txt)
im=Image.open(jpg)
wh=im.size
lbs_new = cxcywh2xyxy(lbs,wh)
print(len(lbs_new), lbs_new)
#print(len(lbs), lbs)

im1 = drawbox(lbs_new,im)
im1.show()

