from mtool.mdraw.tikz.base import Base
from mtool.mdraw.tikz.layers import Layer
from mtool.mdraw.tikz.pathes import Path


def get_head(layers_dire_path):
    return r"""
\documentclass[border=8pt, multi, tikz]{standalone} 
\usepackage{import}
\subimport{""" + layers_dire_path + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image 
\begin{document}
\begin{tikzpicture}
"""


def get_end():
    return r"""
\end{tikzpicture}
\end{document}
"""


def draw_layer(index, chains, module):
    ## 如果没有baseof
    ## 1. 如果是第一个module，则base=(0,0,0)
    ## 2. 否则默认为时前面一个module
    if module.pos_baseof == None:
        if index != 0:
            module.pos_baseof = chains[index - 1]
        else:
            module.pos_baseof = (0, 0, 0)
    return module() + "\n\n"


def draw_path(module, mnames):
    ## 如果path的起点或终点是module，那么先检查是否存在这个module,然后转换成这个module
    if isinstance(module.pos_baseof, str):
        assert module.pos_baseof in mnames, "path:{} module:{} doesn't existed! ".format(module.name, module.pos_baseof)
        module.pos_baseof = mnames[module.pos_baseof]

    if isinstance(module.pos_offset, str):
        assert module.pos_offset in mnames, "path:{} module:{} doesn't existed! ".format(module.name, module.pos_offset)
        module.pos_offset = mnames[module.pos_offset]
    return module() + "\n\n"


def draw_network(layers_dire, chains, tex_path):
    ## 各个模块的名字
    mnames = {}

    ## 生成命令
    command = get_head(layers_dire)
    for index, module in enumerate(chains):
        ## 检查名字是否重用
        assert module.name not in mnames, "module name overleap: [{}]".format(module.name)
        mnames[module.name] = module

        ## 画layer
        if isinstance(module, Layer):
            command += draw_layer(index, chains, module)
            continue

        ## 画path
        if isinstance(module, Path):
            command += draw_path(module, mnames)
            continue

        ## 画其他
        if isinstance(module, Base):
            command += draw_layer(index, chains, module)
            continue

        assert "cannot recognize this module"


    command += get_end()

    print(command)
    with open(tex_path, "w", encoding='UTF-8') as f:
        for c in command:
            f.write(c)
