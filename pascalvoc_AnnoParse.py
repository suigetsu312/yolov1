import xml.etree.ElementTree as ET
import glob
import cv2 as cv
import numpy as np
class object_pascal():
    '''
        this class used to present a object from pascal dataset image
    '''
    def __init__(self,x,y,h,w,category):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.category = category
    def __str__(self):
        return 'x: {}, y: {}, h: {}, w: {}, class :{}'.format(self.x,self.y,self.h,self.w,self.category)

class pascalvocImage():
    def __init__(self, anno_xmlPath, image_basepath):
        self.anno_xmlPath = anno_xmlPath
        #self.samples = []
        self.classes_name = classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
        self.classes_to_id = dict(zip(classes_name,range(len(classes_name))))
        self.sample = []
        self.imagepath = image_basepath + '\\' + self.anno_xmlPath.split('/')[1].split('.')[0] + '.jpg'
        tree = ET.parse(self.anno_xmlPath)
        object_node = tree.findall('object')
        for ob in object_node:
            #get object class
            category = ob.find('name').text
            #get bbox
            bbox = ob.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymin = int(float(bbox.find('ymin').text))
            ymax = int(float(bbox.find('ymax').text))
            
            h = ymax-ymin
            w = xmax-xmin
            x = xmin + w//2
            y = ymin + h//2
            
            self.sample.append(object_pascal(x,y,h,w,category))
    def write_labelFile(self):
        labelFile = 'E:\pascal2012\Data\\' + self.anno_xmlPath.split('\\')[1].split('.')[0] + '.txt'
        with open(labelFile, 'w+') as f:
            #f.writelines('{}\n'.format(self.imagepath))
            for i in self.sample:
                f.writelines('{} {} {} {} {}\n'.format(self.classes_to_id[i.category],i.h,i.w,i.x,i.y))
            

    def resize(self,size):
        '''
        resize image and modify bndbox

        input:
            size : [w,h]
        '''
        sample = self.samples[0]
        img = cv.imread(sample.imagepath)
        new_img = cv.resize(img,size)
        oriSizeX, oriSizeY, _ = np.shape(img)
        
        xscale,yscale = size[0]/oriSizeY, size[1]/oriSizeX
        for idx, sample in enumerate(self.samples):
        
            xmax = sample.x + sample.w//2
            xmin = sample.x - sample.w//2
            ymin = sample.y + sample.h//2
            ymax = sample.y - sample.h//2
            new_x,new_y,new_h,new_w = int(sample.x*xscale),int(sample.y*yscale),int(sample.h*yscale),int(sample.w*xscale)
            sample.x = new_x
            sample.y = new_y
            sample.h = new_h
            sample.w = new_w
            '''
            #CHECK BBOX POSITION 
            cv.rectangle(new_img, (new_xmin, new_ymax), (new_xmax, new_ymin), (0, 255, 0), 2)
            cv.rectangle(new_img, (new_xmin, new_ymin+30), (new_xmin+60, new_ymin), (255, 255, 255), -1)
            cv.putText(new_img, sample.category, (new_xmin, new_ymin+20),
                        cv.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 255), 1, cv.LINE_AA)
            cv.circle(new_img,(new_x,new_y), 3, (255, 0, 0), -1)
            '''

if __name__ == "__main__":
    #take one xml for test
    annoes = glob.glob('E:/pascal2012/VOCdevkit/VOC2012/Annotations/*.xml')
    for anno in annoes:
        t = pascalvocImage(anno,'E:/pascal2012/VOCdevkit/VOC2012/JPEGImages')
        t.write_labelFile()
