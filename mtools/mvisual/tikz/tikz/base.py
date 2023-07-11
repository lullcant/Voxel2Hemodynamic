class Base():
    def __init__(self):
        ## constant
        self.plattle = {
            "conv": "{rgb:yellow,5;red,2.5;white,5}",
            "relu": "{rgb:yellow,5;red,5;white,5}",
            "pool": "{rgb:red,1;black,0.3}",
            "unpool": "{rgb:blue,2;green,1;black,0.3}",
            "softmax": "{rgb:magenta,5;black,7}",
            "arrow": "{rgb:blue,4;red,1;green,1;black,3}",
            "connection-1": "{rgb:blue,4;red,1;green,1;black,3}",
            "connection-2": "{rgb:blue,4;red,1;green,4;black,3}"
        }
        self.box_types = ["Box", "RightBandedBox"]
        self.anchors = ["", "-west", "-east", "-north", "-south", "-anchor", "-near", "-far",
                        "-nearwest", "-neareast", "-farwest", "-fareast",
                        "-northeast", "-northwest", "-southeast", "-southwest",
                        "-nearnortheast", "-farnortheast ",
                        "-nearsoutheast", "-farsoutheast ", "-nearnorthwest",
                        "-farnorthwest", "-nearsouthwest", "-farsouthwest"]

        ## variable
        self.name = ""
        self.opacity = None
        self.draw_color = None
        self.fill_color = None
        self.band_color = None
        self.pos_baseof = None  ## 基准位置 (from)
        self.pos_anchor = None  ## 停靠方向 (default:""，表现是停靠在上一个的东面)
        self.pos_offset = None  ## 偏移位置 (to)

    def check(self):
        '''
        self.pos_baseof 只允许有两种类型：3元tuple / class Base
        self.pos_offset 只允许有两种类型：3元tuple / class Base
        '''
        assert isinstance(self.pos_baseof, Base) or \
               (isinstance(self.pos_baseof, tuple) and len(self.pos_baseof) == 3), \
            "baseof position must be existed module or 3 tuple"

        assert isinstance(self.pos_offset, Base) or \
               (isinstance(self.pos_offset, tuple) and len(self.pos_offset) == 3), \
            "offset position must be existed module or 3 tuple"

        assert self.name != "", "name is empty"
        assert self.pos_anchor in self.anchors, "pos_anchor must be one of anchors"

