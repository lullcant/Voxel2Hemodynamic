

## 得到从src到dst的相对路径
def get_relative_path(src_path, dst_path):
    import os
    '''
    得到从src到dst的相对路径
    :return: 
    '''
    src_abs_path = os.path.abspath(src_path)
    dst_abs_path = os.path.abspath(dst_path)

    index = 0
    common_dire = ""
    for i in range(min(len(src_abs_path), len(dst_abs_path))):
        if src_abs_path[i] == dst_abs_path[i]:
            common_dire += src_abs_path[i]
            index = i
        else:
            break

    rpath = src_abs_path[index + 1:].count('\\') * "../" + \
            dst_abs_path[index + 1:].replace('\\', '/')
    return rpath

## 去除标点符号
def removePunctuation(text, replaced, besides=None):
    '''
    去除标点符号
    :param text: 文本信息
    :param replaced: 替换的字符
    :param besides: 不需要去除的标点符号 []
    :return: newText
    '''
    import string
    temp = []
    for c in text:
        if c in string.punctuation:
            if besides == None:
                temp.append(c)
            else:
                if c in besides:
                    temp.append(c)
                else:
                    temp.append(replaced)
        else:
            temp.append(c)
    newText = ''.join(temp)
    return newText
