import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#导入需要用到的fashion_mnist数据集（因竞赛数据集较大，直接获取）
fashion_mnist = keras.datasets.fashion_mnist
#划分数据
(train_pixel,train_label),(test_pixel,test_label) = fashion_mnist.load_data()
#定义9个分类标签
Target = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
#数据预处理
#至多255个像素点
train_images = train_pixel/255.0
test_images = test_pixel/255.0
#预处理25个数据检查预处理是否完成
plt.figure(figsize=(10,10))
#生成分类图形
for i in range(25):
   plt.subplot(5,5,i+1)
   plt.xticks([])
   plt.yticks([])
   plt.grid(False)
   plt.imshow(train_pixel[i],cmap=plt.cm.binary)
   plt.xlabel(Target[train_label[i]])
plt.show()
#构建神经网络模型
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(128,activation='relu'),keras.layers.Dense(10)])
#编译模型
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
#训练
model.fit(train_pixel,train_label,epochs=10)
#评估
test_loss,test_acc = model.evaluate(test_pixel,test_label,verbose=2)
print('\nTest accuracy:',test_acc)
print('\nTest loss:',test_loss)
#预测
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_pixel)
print(np.argmax(predictions[8]))