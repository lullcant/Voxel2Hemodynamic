from mtool.mdraw.tikz.base import Base


class Layer(Base):
    def __init__(self, name, shapes, position=dict(), args=dict()):
        '''
        定义tikz中的一层
        :param name: 这一层的名字
        :param shapes: 这一层每一块在图像上显示的,默认单位：pt, [list] example: [[2,2],8,8]
                       widths: 宽度，注意，考虑到多块的可能，宽度这里传入的应是数组
                       height: 高度
                       depth:  深度
        :param position: 这一层的位置 [dict]
                       baseof: 基础位置  (默认为上一个module，如果没有上一个module则默认为(0,0,0))
                       anchor: 相对方向  (默认为上一个东面，仅当offset=(0,0,0)时有效)
                       offset: 偏移距离  (必须是3元tuple)
        :param args: [dict]
            :key label: 图像上的标注 [dict]
                       xlabel: 宽度方向上的标注，考虑到多块的可能，这里传入的应是数组
                       ylabel: 高度方向上的标注
                       zlabel: 深度方向上的标注
                       caption: 图像的标注
            :key front: 字体大小 [list]
                       frontsize: 字号
                       frontspace: 字体间隔
        '''
        super(Layer, self).__init__()

        ## basic variable
        self.name = name
        self.position = position
        self.widths, self.height, self.depth = shapes
        self.pos_baseof = None
        self.pos_anchor = ""
        self.pos_offset = (0, 0, 0)

        ## 其他参数
        self.args = args

    def check(self):
        super(Layer, self).check()
        assert isinstance(self.pos_offset, tuple) and len(self.pos_offset) == 3, "offset must be 3 tuple"

    ## 字体
    def get_front(self):
        if "front" in self.args:
            frontsize, frontspace = self.args["front"]
            str_front = ",font=\\fontsize{{{0}}}{{{1}}}\selectfont".format(frontsize, frontspace)
            return str_front
        return ""

    ## 间隔
    def get_space(self):
        return "  "

    ## 名字
    def get_name(self):
        str_name = self.get_space() * 4 + "name={name},\n".format(name=self.name)
        return str_name

    ## 颜色：fill, bandfill, opacity,
    def get_color(self):
        assert "no clolr"
        pass

    ## 样式
    def get_boxtype(self):
        assert "no type"

    ## 大小
    def get_shape(self):
        str_width = self.get_space() * 4 + "width={{{}}},\n".format(str(self.widths).replace('[', '').replace(']', ''))
        str_hight = self.get_space() * 4 + "height={},\n".format(self.height)
        str_depth = self.get_space() * 4 + "depth={},\n".format(self.depth)
        return str_width + str_hight + str_depth

    ## 标签
    def get_label(self):
        if "label" in self.args:
            command = ""
            label = self.args["label"]
            for key in label:
                if key == "xlabel":
                    assert isinstance(label[key],list), "xlabel must be list"
                    command += self.get_space() * 4 + "xlabel={{{}}},\n".format(
                        str(label[key]).replace('[', '{').replace(']', '}'))
                    continue
                command += self.get_space() * 4 + "{key}={value},\n".format(key=key, value=label[key])
            return command
        return ""

    ## 位置
    def get_position(self):
        if "baseof" in self.position: self.pos_baseof = self.position["baseof"]
        if "anchor" in self.position: self.pos_anchor = self.position["anchor"]
        if "offset" in self.position: self.pos_offset = self.position["offset"]

    ## 头部，主要判断位置
    def get_head(self):
        self.get_position()
        command = "\pic[shift={{{shift}}}{front}] at {baseof}{direction}\n".format(
            shift=self.pos_offset,
            front=self.get_front(),
            baseof=self.pos_baseof
            if isinstance(self.pos_baseof, tuple)
            else "({})".format(self.pos_baseof.name),
            direction=self.pos_anchor
        )
        return command

    ## 盒子
    def get_box(self):
        args = "{name}{shape}{color}{label}".format(
            name=self.get_name(),
            shape=self.get_shape(),
            color=self.get_color(),
            label=self.get_label()
        )

        command = self.get_space() * 2 + "{{{box}={{\n{args}".format(box=self.get_boxtype(), args=args) \
                  + self.get_space() * 2 + "}\n};"
        return command

    def __call__(self):
        self.check()
        command = self.get_head() + self.get_box()
        return command


class dconv(Layer):
    def __init__(self, name, shapes, position=dict(), args=dict()):
        super(dconv, self).__init__(name, shapes, position, args)

    def get_color(self):
        str_fill = self.get_space() * 4 + "fill={},\n".format(self.plattle['conv'])
        return str_fill

    def get_boxtype(self):
        return self.box_types[0]


class dconvb(Layer):
    def __init__(self, name, shapes, position=dict(), args=dict()):
        super(dconvb, self).__init__(name, shapes, position, args)

    def get_color(self):
        str_fill = self.get_space() * 4 + "fill={},\n".format(self.plattle['conv'])
        str_bandfill = self.get_space() * 4 + "bandfill={},\n".format(self.plattle['relu'])
        return str_fill + str_bandfill

    def get_boxtype(self):
        return self.box_types[1]


class dpool(Layer):
    def __init__(self, name, shapes, position=dict(), args=dict()):
        super(dpool, self).__init__(name, shapes, position, args)

    def get_color(self):
        str_fill = self.get_space() * 4 + "fill={},\n".format(self.plattle['pool'])
        str_opacity = self.get_space() * 4 + "opacity={0.5},\n"
        return str_fill + str_opacity

    def get_boxtype(self):
        return self.box_types[0]


class dunpool(Layer):
    def __init__(self, name, shapes, position=dict(), args=dict()):
        super(dunpool, self).__init__(name, shapes, position, args)

    def get_color(self):
        str_fill = self.get_space() * 4 + "fill={},\n".format(self.plattle['unpool'])
        str_opacity = self.get_space() * 4 + "opacity={0.5},\n"
        return str_fill + str_opacity

    def get_boxtype(self):
        return self.box_types[0]
