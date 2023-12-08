import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
class PascalDataLoader:
    def __init__(self, dataset_path, batch_size, size=[224, 224], mode=0):
        # ... (與之前相同的初始化代碼)
        self.dataset_path = dataset_path
        self.image_path = dataset_path + '\\VOCdevkit\\VOC2012\\JPEGImages\\'
        self.txt_sample_path = dataset_path + '\\Data\\'
        self.mode = mode
        if mode == 0:
            d_set_name = '\\train.txt'
        elif mode == 1:
            d_set_name = '\\test.txt'
        else:
            raise ValueError(f'mode must be 0 or 1')
        self.data_set = open(self.dataset_path + d_set_name, 'r').read().split('\n')[:-1]
        self.datasetLen = len(self.data_set)
        random.shuffle(self.data_set)
        self.S = 7
        self.B = 2
        self.size = size
        self.classes = 20
        self.cell_len = np.array([x/self.S for x in self.size])
        self.batch_size = batch_size

    def read(self, fname):
        def split_object(line):
            return [float(x) for x in line.split(' ')]

        sample = cv.imread(fname.numpy().decode()).astype(np.float32)
        sample = cv.resize(sample, (224, 224))

        label = np.zeros([self.S, self.S, 5*self.B+self.classes])
        with open(self.txt_sample_path + fname.numpy().decode().split('\\')[-1].split('.')[0] + '.txt', mode='r') as f:
            objects = [split_object(x) for x in f.readlines()]
        objects = np.stack(objects, axis=0)
        for o in objects:
            classes = o[0].astype(np.int8)
            xy = np.array(o[1:3])
            wh = np.array(o[3:5])
            ss = np.array([self.S, self.S])
            grid_index = np.floor(xy * ss).astype(np.int8)
            label[grid_index[1], grid_index[0], classes] = 1
            label[grid_index[1], grid_index[0], 20:22] = xy*ss - grid_index
            label[grid_index[1], grid_index[0], 22:24] = np.log(wh*7)
            label[grid_index[1], grid_index[0], 24:26] = xy*ss - grid_index
            label[grid_index[1], grid_index[0], 26:28] = np.log(wh*7)
            label[grid_index[1], grid_index[0], 28] = 1
            label[grid_index[1], grid_index[0], 29] = 1

        return sample, label

    def create_dataset(self):
        filenames = [fname for fname in self.data_set]
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(lambda x: tf.py_function(self.read, [x], [tf.float32, tf.float32]),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
    
    def get_sample_by_filename(self, filename):
        return self.read(filename)

    def draw_bounding_box_by_filename(self, filename):
        sample, label = self.get_sample_by_filename(filename)
        draw_bounding_box(sample, label)

def draw_bounding_box(image, label):
    # 由於 label 的形狀是 (S, S, 5 * B + classes)，需要進行一些轉換以獲得坐標和尺寸信息
    classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    S = label.shape[0]
    cell_size = image.shape[0] // S
    for i in range(S):
        for j in range(S):
            # 如果該單元格的第 28 個元素（最後一個元素）為 1，表示這個單元格包含對象
            if label[i, j, 28] == 1:
                # 獲取預測的 xywh 坐標和 wh
                xywh = label[i,j,20:24]
                c = -1
                index = 0
                for labelC in label[i, j, :20]:

                    print(labelC, c)
                    if labelC == 1:
                        c = index
                        break
                    index +=1
                xy = (xywh[:2] + np.array([j,i]))/7
                wh = np.exp(xywh[2:])/7
                x = xy[0]
                y = xy[1]
                w = wh[0]
                h = wh[1]
                xmin = int((x-w/2)*image.shape[1])
                ymin = int((y-h/2)*image.shape[0])
                xmax = int((x+w/2)*image.shape[1])
                ymax = int((y+h/2)*image.shape[0])
                if c > -1:
                    # 繪製 bounding box
                    cv.putText(image, str(classes_name[int(c)]), (xmin, ymin+20),
                        cv.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), 1, cv.LINE_AA)
                image = cv.rectangle(image, (xmin,ymin), (xmax,ymax), color=(0, 255, 0), thickness=2)

    # 將 BGR 圖像轉換為 RGB
    image_rgb = cv.cvtColor(image.astype(np.uint8), cv.COLOR_BGR2RGB)
    
    # 顯示圖像
    plt.imshow(image_rgb)
    plt.show()
if __name__ == "__main__":
    # 測試函數
    dataset_path = 'D:\\programming\\MLDL\\dataset\\pascal2012'
    specific_filename = 'D:\\programming\\MLDL\\dataset\\pascal2012\\VOCdevkit\\VOC2012\\JPEGImages\\2008_004321.jpg'

    batch_size = 1  # 設置為 1，因為我們只想測試單個樣本
    test_gen = PascalDataLoader(dataset_path, batch_size, mode=0)
    test_dataset = test_gen.create_dataset()

    # 從數據集中獲取一個樣本
    sample, label = next(iter(test_dataset))

    # 調用繪製 bounding box 的函數
    #draw_bounding_box(sample[0].numpy(), label[0].numpy())
    test_gen.draw_bounding_box_by_filename(specific_filename)
