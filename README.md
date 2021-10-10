## 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

## 修改train.py中的数据集
利用 self.add_class来添加自己数据集，设定前确定mask png图像中每个像素值对应的类，如：1对应类“grasper",那么我们添加：
self.add_class("shapes", 1, "grasper")

## 修改完直接训练
python train.py

## 运行demo
首先修改训练后模型.h5文件的路径；
其次修改：class_names 和 image_path
运行：python demo.py

## Tips:
train.py中相关参数在 ShapesConfig 类中设置，如batch_size 和 learning_rate