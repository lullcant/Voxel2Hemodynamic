from mtools.mvisual.mplot import plot_3d_points
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math


class Graph:
    def __init__(self, num_verts, vertices=None):
        '''
        :param vertices: the number of vertices
        :param nodes:
        '''
        # Number of vertices
        self.num_verts = num_verts
        if vertices is not None:
            assert isinstance(vertices, list), 'vertices should be a list'
            self.vertices = vertices.copy()
        else:
            self.vertices = [i for i in range(self.num_verts)]

        # default dictionary to store graph
        self.graph = {index: set() for index in range(self.num_verts)}

    # function to add an edge to graph
    def add_edge(self, u, v, is_index=True, dual=True):
        '''
        :param u: the end of the edge
        :param v: the other end of the edge
        :param is_index: True  -> u,v is the index of the vertice /
                         False -> u,v is the value of the vertice
        :param dual: True -> undirected graph / False -> directed graph
        :return:
        '''
        assert u != v, 'u !=v in Graph'

        if is_index:
            uindex, vindex = u, v
        else:
            assert u in self.vertices, "{} is not in vertices".format(u)
            assert v in self.vertices, "{} is not in vertices".format(v)
            uindex, vindex = self.vertices.index(u), self.vertices.index(v)

        self.graph[uindex].add(vindex)
        if dual:
            self.graph[vindex].add(uindex)

    def del_edge(self, u, v, is_index=True, dual=True):
        assert u != v, 'u !=v in Graph'
        if is_index:
            uindex, vindex = u, v
        else:
            assert u in self.vertices, "{} is not in vertices".format(u)
            assert v in self.vertices, "{} is not in vertices".format(v)
            uindex, vindex = self.vertices.index(u), self.vertices.index(v)

        self.graph[uindex].remove(vindex)
        if dual:
            self.graph[vindex].remove(uindex)

    ## function to get all edges
    def get_edges(self, is_index=True):
        '''
        :param is_index: True   -> get the index of two points in an edge
                         False  -> get the value of two points in an edge
        :return:
        '''
        edges = set()
        for uindex in self.graph:
            for vindex in self.graph[uindex]:
                edges.add((uindex, vindex))

        edges = list(edges)
        result = []
        if not is_index:
            for edge in edges:
                result.append([self.vertices[edge[0]], self.vertices[edge[1]]])
        else:
            result = edges

        return result

    def get_verts(self):
        return self.vertices

    def get_num_verts(self):
        return self.num_verts

    ## Deep first search
    def dfs(self, start, is_index=True):
        '''
        :return: the path of deepest path
        '''

        if not is_index:
            assert start in self.vertices, "{} is not in vertices".format(start)
            start = self.vertices.index(start)

        ## the vertices have been visited
        visited = list()
        longest = (0, [])

        def dfs_traverse(v):
            nonlocal longest
            ## this vertice has been visited
            visited.append(v)

            ## connection vertices
            verts = list(self.graph[v])
            verts = [v for v in verts if v not in visited]
            # print("visited: {} verts: {}".format(visited,verts))

            ## Cannot go on deeper search
            if len(verts) == 0:
                seq = visited.copy()
                if len(seq) > longest[0]:
                    longest = (len(seq), seq)
                visited.pop()
                return

            ## Traverse All connected vertices
            for vert in verts:
                dfs_traverse(vert)
            visited.pop()

        dfs_traverse(start)
        return longest

    def bfs(self, start, end=None, is_index=True):
        '''
        :param start:
        :param end:
        :param is_index:
        :return: end = None 时，返回遍历到所有终点的路径
        '''
        current = [start]
        visired = set()
        ## 所有遍历的路径
        travsed = [[start]]
        ## 所有遍历的路径的末端
        pthends = [start]

        ## 从start到所有叶子节点的路径
        pathes = []
        while len(current) > 0:
            vert = current.pop(0)
            visired.add(vert)
            next = list(self.graph[vert])
            path = travsed[pthends.index(vert)]

            isnext = False
            for idx in next:
                if idx in visired:
                    continue
                isnext = True
                current.append(idx)

                if end is not None and idx in end:
                    return path + [idx]

                travsed.append(path + [idx])
                pthends.append(idx)

            if not isnext:
                pathes.append(path)

        return pathes

    # def is_connected(self,start, end):
    #     '''
    #     :param start: 起点
    #     :param end: 终点
    #     :return:
    #     '''
    #






    ## remove empty vertice
    def remove_empty(self):

        ## 找到点的对应转换
        trans = {}
        verts = []
        for idx, vert in enumerate(self.vertices):
            if len(vert) == 0:
                continue
            trans[idx] = len(verts)
            verts.append(vert)

        newgraph = Graph(len(verts), vertices=verts)
        ## 添加边
        for u_vert, edges in self.graph.items():
            if len(edges) == 0:
                continue

            for v_vert in edges:
                newgraph.add_edge(
                    u=trans[u_vert],
                    v=trans[v_vert]
                )
        return newgraph

    ## build tree from points
    def build_tree(self, num_segment=1):
        ## 分支合并的思想，把每一个想成一个段，逐渐将最近的段段之间相连

        ## 计算这个段block和其他剩余段离的最近距离
        def corrd_min_dist(block, segments):
            min_dists = math.inf
            min_sgidx = -1
            min_ptidx = None

            for index, segment in enumerate(segments):
                dist = cdist(XA=np.asarray(list(np.asarray(self.vertices)[block])),
                             XB=np.asarray(list(np.asarray(self.vertices)[segment])))
                dmin = np.min(dist)

                if dmin < min_dists:
                    min_dists = dmin
                    min_sgidx = index
                    row, col = np.where(dist == dmin)
                    min_ptidx = [row[0], col[0]]

            return min_sgidx, min_ptidx

        ## 初始化，将每个通过edges连接起来的点集作为一个分段，如果没有连接就各自成为一段
        svisited = set()
        segments = []

        for vert, edges in self.graph.items():
            if vert in svisited or len(self.vertices[vert]) == 0:
                continue

            connects = self.bfs(start=vert)[0]
            segments.append(connects)
            svisited = svisited.union(set(connects))

        ## 开始合并段
        while True:
            if len(segments) == num_segment:
                break

            sgid, ptid = corrd_min_dist(segments[0], segments[1:])

            ## 建立连接
            self.add_edge(u=segments[0][ptid[0]], v=segments[1:][sgid][ptid[1]])

            ## 合并分段
            new_elem = segments[0] + segments[1:][sgid]
            segments.pop(1 + sgid)
            segments.pop(0)
            segments.append(new_elem)

        ## 树建立完成

    def get_tree(self, start=None):
        ## note: do build_tree first !
        ## dfs: get branches

        tree = []

        def dfs(start, head=None):
            p = start
            if head is not None:
                branch = [head, p]
            else:
                branch = [p]
            while len(self.graph[p].difference(set(branch))) == 1:
                p = list(self.graph[p].difference(set(branch)))[0]
                branch.append(p)

            tree.append(branch)
            if len(self.graph[p].difference(set(branch))) > 1:
                heads = list(self.graph[p].difference(set(branch)))
                for head in heads:
                    dfs(start=head, head=p)

        if start is not None:
            dfs(start)
        else:
            dfs(self.get_key_points('end')[0])

        return tree

    def get_acyclic_graph(self):
        '''
        dfs, if v has been visited, there is a circle
        :return:
        '''

        acyclic_graph = Graph(self.num_verts, self.vertices)

        ## the vertices have been visited
        visited = list()
        abandon = list()

        def dfs_traverse(v, last):
            nonlocal abandon
            ## this vertice has been visited
            visited.append(v)

            ## connection vertices
            verts = list(self.graph[v])
            abandon += [[v, vert] for vert in verts if vert in visited and vert != last]
            verts = [vert for vert in verts if vert not in visited]
            # print("v: {} verts: {} visited:{}".format(v,verts,visited))

            ## Cannot go on deeper search
            if len(verts) == 0:
                visited.pop()
                return

            ## Traverse All connected vertices
            for vert in verts:
                ## all connected and not visited
                acyclic_graph.add_edge(u=v, v=vert)
                if [vert, v] in abandon or [v, vert] in abandon:
                    continue
                dfs_traverse(vert, v)
            visited.pop()

        dfs_traverse(0, None)

        for edge in abandon:
            acyclic_graph.del_edge(u=edge[0], v=edge[1])

        return acyclic_graph

    def get_prunned_graph(self, threshold, excepts=None):
        ins_points = self.get_key_points(mode='ins')
        end_points = self.get_key_points(mode='end')
        if excepts is not None:
            end_points = list(set(end_points) - set(excepts))

        key_points = ins_points + end_points

        ## immediately return once hit the key point
        def dfs_hit(start):
            ## connected points from end to end/ins
            connect_points = list()

            ## the vertices have been visited
            visited = list()

            def dfs_traverse(v):
                connect_points.append(v)
                if v in key_points and v != start:
                    return

                ## this vertice has been visited
                visited.append(v)

                ## connection vertices
                verts = list(self.graph[v])
                verts = [v for v in verts if v not in visited]

                ## Cannot go on deeper search
                if len(verts) == 0:
                    visited.pop()
                    return

                ## Traverse All connected vertices
                for vert in verts:
                    dfs_traverse(vert)
                visited.pop()

            dfs_traverse(start)
            return connect_points

        ## traverse all end point
        abandon_points = []
        for pnt in end_points:
            connect_points = dfs_hit(pnt)
            ## prun short section
            if len(connect_points) < threshold:
                abandon_points += list(set(connect_points) - set(ins_points))

        reserved_points = list(set([i for i in range(self.num_verts)]) - set(abandon_points))
        values = [self.vertices[i] for i in reserved_points]

        prunned_graph = Graph(num_verts=len(reserved_points), vertices=values)

        edges = self.get_edges()
        for edge in edges:
            u, v = edge[0], edge[1]
            if u in reserved_points and v in reserved_points:
                prunned_graph.add_edge(u=reserved_points.index(u), v=reserved_points.index(v))

        return prunned_graph

    ## get key points of the current graph (skeleton)
    def get_key_points(self, mode='all', is_index=True):
        '''
        :param dis_func:   calculate the distance between two node of the graph   (default: None)
        :param threshold:  if < distance, the point will be judge as the same one (default: None)
        :param mode: 'all' -> (intersection point + end point) [default]
                     'end' -> (end point)
                     'ins' -> (intersection point)
        :return:
        '''
        result = []
        for v in self.graph:
            if (mode == 'all' or mode == 'ins') and len(self.graph[v]) > 2:
                result.append(v)

            if (mode == 'all' or mode == 'end') and len(self.graph[v]) == 1:
                result.append(v)

        if not is_index:
            result = [self.vertices[i] for i in result]

        return result

    ## get key graph of the current graph (skeleton)
    def get_key_graph(self):
        key_points = self.get_key_points(mode='all')

        ## immediately return once hit the key point
        def dfs_hit(start):
            ## connected points
            connect_points = set()

            ## the vertices have been visited
            visited = list()

            def dfs_traverse(v):
                if v in key_points and v != start:
                    connect_points.add(v)
                    return

                ## this vertice has been visited
                visited.append(v)

                ## connection vertices
                verts = list(self.graph[v])
                verts = [v for v in verts if v not in visited]

                ## Cannot go on deeper search
                if len(verts) == 0:
                    visited.pop()
                    return

                ## Traverse All connected vertices
                for vert in verts:
                    dfs_traverse(vert)
                visited.pop()

            dfs_traverse(start)
            return list(connect_points)

        key_graph = Graph(len(key_points), [self.vertices[i] for i in key_points])
        for uindex, pnt in enumerate(key_points):
            connected_points = dfs_hit(pnt)
            for cp in connected_points:
                vindex = key_points.index(cp)
                key_graph.add_edge(u=uindex, v=vindex)

            # for cp in connected_points:
            #     vindex = key_points.index(cp)
            #     key_graph.add_edge(u=uindex, v=vindex)

        return key_graph

    def visual_graph(self):
        ## 展示树
        verts = self.get_verts()
        edges = self.get_edges()

        ax = plot_3d_points(verts, end=False, color='blue')

        # fig = plt.figure()
        # ax = Axes3D(fig)

        for u, v in edges:
            x = [verts[u][0], verts[v][0]]
            y = [verts[u][1], verts[v][1]]
            z = [verts[u][2], verts[v][2]]

            ax.plot(x, y, z, color='red')

        ends = self.get_key_points('end')
        ints = self.get_key_points('ins')
        points = np.asarray(self.get_verts())

        plot_3d_points(points[ends], ax=ax, end=False, color='orange', size=40)
        plot_3d_points(points[ints], ax=ax, color='green', size=40)

    def save_graph(self, points_path, edges_path):
        np.save(points_path, self.vertices)
        np.save(edges_path, self.get_edges())

    @staticmethod
    def load_graph(points_path, edges_path):
        vertices = np.load(points_path)
        graph = Graph(len(vertices), vertices.tolist())
        edges = np.load(edges_path)
        for edge in edges:
            graph.add_edge(u=edge[0], v=edge[1])

        return graph

    ## show graph
    def __repr__(self):
        str = ""
        for node in self.graph:
            str += "[{}]->{}\n".format(node, self.graph[node])
        return str


def example_get_tree():
    points = np.load('../support/tree-points.npy')
    graph = Graph(num_verts=len(points), vertices=points.tolist())
    graph.build_tree()
    graph = graph.get_prunned_graph(threshold=3)

    ## 展示树
    verts = graph.get_verts()
    edges = graph.get_edges()
    # ax = plot_3d_points(verts, end=False, color='blue')

    fig = plt.figure()
    ax = Axes3D(fig)

    for u, v in edges:
        x = [verts[u][0], verts[v][0]]
        y = [verts[u][1], verts[v][1]]
        z = [verts[u][2], verts[v][2]]

        ax.plot(x, y, z, color='red')

    ends = graph.get_key_points('end')
    ints = graph.get_key_points('ins')
    points = np.asarray(graph.get_verts())

    plot_3d_points(points[ends], ax=ax, end=False, color='orange', size=40)
    plot_3d_points(points[ints], ax=ax, color='green', size=40)

    branch = graph.get_tree()
    points = np.asarray(graph.get_verts())
    fig = plt.figure()
    ax = Axes3D(fig)
    for bn in branch:
        plot_3d_points(points[bn], ax=ax, end=False)
    plt.show()


def example_get_tree_2(path='./mtools/support/centerline.json'):
    from mtools.mio import get_json, save_csv

    xyz = []
    json = get_json(path)['PipeNodes']
    for segment in json:
        if len(segment['Path']) == 0:
            continue

        path = segment['Path']
        for idx in range(len(path) // 3):
            xyz.append(np.round(np.asarray(path[idx * 3:idx * 3 + 3]) / 2., decimals=2))
        xyz.append([])

    # save_csv('./centerline.xyz', rows=xyz, delimiter=' ')
    # exit()

    graph = Graph(num_verts=len(xyz), vertices=xyz)
    for idx, point in enumerate(xyz):
        if len(point) == 0 or len(xyz[idx + 1]) == 0:
            continue

        graph.add_edge(u=idx, v=idx + 1)

    graph = graph.remove_empty()
    graph.build_tree(num_segment=2)
    graph = graph.get_prunned_graph(threshold=3)
    # graph.show_tree()

    xyz = []
    tree = graph.get_tree()
    for branch in tree:
        for idx in branch:
            xyz.append(graph.vertices[idx])
        xyz.append([])
    save_csv('./left.xyz', rows=xyz, delimiter=' ')

    start = list(set(graph.get_key_points('end')) - set(list(np.concatenate(tree, axis=0))))[0]
    xyz = []
    tree = graph.get_tree(start=start)
    for branch in tree:
        for idx in branch:
            xyz.append(graph.vertices[idx])
        xyz.append([])
    save_csv('./right.xyz', rows=xyz, delimiter=' ')


if __name__ == '__main__':
    example_get_tree()

    exit()
