# ReadMe




## IO

### 读取 DICOM 格式数据

```python
def get_dicom_image(dire):
    '''
    加载一组dicom序列图像
    :param dire: dicom序列所在的文件夹路径，E.g. "E:/Work/Database/Teeth/origin/1/"
    :return: (array,origin,spacing,direction)
    array:  图像数组
    origin: 三维图像坐标原点
    spacing: 三维图像坐标间距
    direction: 三维图像坐标方向
    注意：实际取出的数组不一定与MITK或其他可视化工具中的方向一致！
    可能会出现旋转\翻转等现象，这是由于dicom头文件中的origin,spacing,direction的信息导致的
    在使用时建议先用matplotlib.pyplot工具查看一下切片的方式是否异常，判断是否需要一定的预处理
    注意：实际DICOM第一张可能是定位图，同时取出会导致位置错乱
    '''
```



### 读取其他格式
