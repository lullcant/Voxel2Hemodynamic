import os

from mtool.mdraw.tikz import tikzlayers
from mtool.mdraw.tikz.construct import *
from mtool.mdraw.tikz.layers import *
from mtool.mdraw.tikz.pathes import *
from mtool.mdraw.tikz.other import *
from mtool.mutils.mutils import get_relative_path


def main():
    tex_path = "./unet.tex"

    ##  8-16-25-32-40
    chains = [
        dimage(name='input', shapes=[8, 8],
               image_path=get_relative_path(tex_path, "./tikz/add-image/miao.png")),

        dconvb(name="conv1x1", shapes=[[4], 8, 8], position={'offset': (1, 0, 0)}),

        dunpool(name="l1-unpool", shapes=[[1], 16, 16], position={'offset': (1, 0, 0)}),
        dconvb(name="l1-conv", shapes=[[2, 2, 4], 16, 16], args={"label": {"xlabel": [4, 4, 8], "zlabel": 1024}}),

        dunpool(name="l2-unpool", shapes=[[1], 16, 16], position={'offset': (1, 0, 0)}),
        dconvb(name="l2-conv", shapes=[[2, 2, 4], 25, 25], args={"label": {"xlabel": [4, 4, 8], "zlabel": 2048}}),

        skip(name="s1",mfrom="l1-conv",mto="l2-unpool",pos=1.5),

    ]

    draw_network(layers_dire=get_relative_path(tex_path, os.path.dirname(tikzlayers.__file__)),
                 tex_path=tex_path,
                 chains=chains)


if __name__ == '__main__':
    main()
