from mtool.mdraw.tikz.base import Base


class dimage(Base):
    def __init__(self, name, shapes, image_path, position=dict()):
        super(dimage, self).__init__()
        ## basic variable
        self.name = name
        self.position = position
        self.height, self.depth = shapes
        self.image_path = image_path

    ## 位置
    def get_position(self):
        self.str_offset = ""
        if "baseof" in self.position: self.pos_baseof = self.position["baseof"]
        if "offset" in self.position:
            self.str_offset = ",shift={{{}}}".format(self.position["offset"])

    def __call__(self):
        self.get_position()

        command = "\\node[canvas is zy plane at x=0{shift}] ({name}) at {baseof}" \
                  "{{\includegraphics[width={height}cm,height={depth}cm]{{{path}}}}};" \
            .format(name=self.name,
                    shift=self.str_offset,
                    baseof=self.pos_baseof
                    if isinstance(self.pos_baseof, tuple)
                    else "({})".format(self.pos_baseof.name),
                    height=self.height / 5, depth=self.depth / 5,
                    path=self.image_path)
        return command
