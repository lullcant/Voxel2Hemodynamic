from mtool.mdraw.tikz.base import Base
from mtool.mdraw.tikz.layers import Layer


class Path(Base):
    def __init__(self, name, mfrom, mto, args=dict()):
        '''
        :param mfrom: 路径起点
        :param mto:   路径终点
        :param args: [dict]
            :key color: 线段颜色
            :key anchors: 起始点和终点的停靠方向，仅当mfrom/mto是module时使用 dict() {"from":"","to":""}
            :key opacity: 路径透明度
            :key line_width: 线段宽度
        :return:
        '''
        super(Path, self).__init__()

        ## name
        self.name = name

        ## 起始点路径
        self.pos_baseof = mfrom
        self.pos_offset = mto
        self.anchor_baseof = "-east"
        self.anchor_offset = "-west"

        ## 参数
        self.args = args

    def get_position(self):
        if not isinstance(self.pos_baseof, Path):
            if "anchors" in self.args and "from" in self.args["anchors"]:
                self.anchor_baseof = self.args["anchors"]["from"]
            self.pos_baseof = "({}{})".format(self.pos_baseof.name, self.anchor_baseof)

        if not isinstance(self.pos_offset, Path):
            if "anchors" in self.args and "to" in self.args["anchors"]:
                self.anchor_offset = self.args["anchors"]["to"]
            self.pos_offset = "({}{})".format(self.pos_offset.name, self.anchor_offset)

        if (isinstance(self.pos_baseof, tuple) and len(self.pos_baseof) == 3):
            self.pos_baseof = "{}".format(self.pos_baseof)

        if (isinstance(self.pos_offset, tuple) and len(self.pos_offset) == 3):
            self.pos_offset = "{}".format(self.pos_offset)

    def get_color(self):
        self.color = "{rgb:blue,4;red,1;green,1;black,3}"
        if "color" in self.args:
            self.color = self.args["color"]
        self.opacity = 0.7
        if "opacity" in self.args:
            self.opacity = self.args["opacity"]
        return "draw={},opacity={}".format(self.color, self.opacity)

    def get_linewidth(self):
        self.line_width = 1
        if "line_width" in self.args:
            self.line_width = self.args["line_width"]
        return "line width={},".format(self.line_width)

    def get_node(self):
        return ""

    def __call__(self):
        self.get_position()

        command = "\draw[{width}, every node/.style={{sloped, allow upside down}},{color}]\n". \
                      format(width=self.get_linewidth(), color=self.get_color()) + \
                  "{mfrom} -- {node}{mto};".format(mfrom=self.pos_baseof, mto=self.pos_offset, node=self.get_node())

        return command


class connect(Path):
    def __init__(self, name, mfrom, mto, args=dict()):
        super(connect, self).__init__(name, mfrom, mto, args=args)

    def get_color(self):
        self.color = "{rgb:blue,4;red,1;green,4;black,3}"
        if "color" in self.args:
            self.color = self.args["color"]
        self.opacity = 0.7
        if "opacity" in self.args:
            self.opacity = self.args["opacity"]
        return "draw={},opacity={}".format(self.color, self.opacity)

    def get_node(self):
        return "node {{ \\tikz \draw[-Stealth,{line_width}{color}] (-0.3,0) -- ++(0.3,0);}}". \
            format(line_width=self.get_linewidth(), color=self.get_color())


class skip(Path):
    def __init__(self, name, mfrom, mto, pos, args=dict()):
        super(skip, self).__init__(name, mfrom, mto, args=args)
        self.pos = pos
        self.anchor_baseof = ""
        self.anchor_offset = ""

    def get_position(self):
        if isinstance(self.pos_baseof, Layer):
            self.pos_baseof = self.pos_baseof.name

        if isinstance(self.pos_offset, Layer):
            self.pos_offset = self.pos_offset.name

        if (isinstance(self.pos_baseof, tuple) and len(self.pos_baseof) == 3):
            assert "mfrom can't be 3 tuple in skip"

        if (isinstance(self.pos_offset, tuple) and len(self.pos_offset) == 3):
            assert "mto can't be 3 tuple in skip"

    def get_node(self):
        return "node {{ \\tikz \draw[-Stealth,{line_width}{color}] (-0.3,0) -- ++(0.3,0);}}". \
            format(line_width=self.get_linewidth(), color=self.get_color())

    def get_color(self):
        self.color = "{rgb:blue,4;red,1;green,1;black,3}"
        if "color" in self.args:
            self.color = self.args["color"]
        self.opacity = 0.7
        if "opacity" in self.args:
            self.opacity = self.args["opacity"]
        return "draw={},opacity={}".format(self.color, self.opacity)

    def __call__(self):
        self.get_position()

        command = "\path ({mfrom}-southeast) -- ({mfrom}-northeast) coordinate[pos={pos}] ({mfrom}-top);\n" \
                  "\path ({mto}-south) -- ({mto}-northeast) coordinate[pos={pos}] ({mto}-top);\n" \
                  "\draw[{width}, every node/.style={{sloped, allow upside down}},{color}] ({mfrom}-northeast)\n" \
                  "-- {node}({mfrom}-top)\n" \
                  "-- {node}({mto}-top)\n" \
                  "-- {node}({mto}-north);". \
            format(mfrom=self.pos_baseof, mto=self.pos_offset, pos=self.pos,
                   width=self.get_linewidth(), color=self.get_color(), node=self.get_node())

        return command

