import operator
import math
import random
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy import stats

import input_data
# import Spline
import Spline2

MIN_DEPTH = 2  # 初期個体の最小の木の深さ
MAX_DEPTH = 5  # 初期個体の最大の木の深さ


class Tree(list):
    def __init__(self,content):
        list.__init__(self,content)
        self.fitness = None
    @property
    def height(self):
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth+1] * elem[2])
        return max_depth
    def searchSubtree(self, begin):
        """交叉に使う部分木を生成

        Args:
            begin (int): 部分木の根ノード.もともとの個体（木）のインデックス番号で表す．（どこから木を切って部分木を作るか）
            end (int): 部分木の最後の葉ノード．もともとの個体（木）のインデックス番号で表す．（どこまで木を切るか）

        Returns:
            slice object(): slice(begin, end)オブジェクトを返す．
        """
        end = begin + 1
        total = self[begin][2]
        while total > 0:
            total += self[end][2] - 1
            end += 1
        return slice(begin, end)
    def __str__(self):
        start = 0
        str_arr = [""]
        for node in self:
            for i in range(len(str_arr)):
                if str_arr[i] == "":
                    name,func,arity = node
                    if arity != 0:
                        tmp_arr = [name,"(",""]
                        for x in range(arity-1):
                            tmp_arr += [",",""]
                        tmp_arr.append(")")
                        str_arr = str_arr[:i] + tmp_arr + str_arr[i+1:]
                    else:
                        str_arr = str_arr[:i] + [name] + str_arr[i+1:]
                    break
        return "".join(str_arr)


def generate_dataset(dataset):
    all_data = []
    data = []
    knots = []
    for i, j in zip(dataset[:, 1], dataset[:, 0]):
        all_data.append([i-2.5, j])
        data.append([i-2.5, j])
    for i, j in zip(Spline2.opt_knots, Spline2.opt_spline.predict(Spline2.opt_knots)):
    # for i, j in zip(Spline.new_knots, Spline.new_y_est): # スプライン節点座標のデータを追加
        all_data.append([i, j])
        knots.append([i, j])
    # knots.append([100, 0])
    # knots.insert(0, [0, 0])
    data, knots = np.array(data), np.array(knots)
    sorted_data = np.array(sorted(all_data, key=lambda x: x[0]))
    return data[:,0], data[:, 1], knots[:, 0], knots[:, 1], sorted_data
    # return sorted_data[:, 0], sorted_data[:, 1]

def calc_var(num):
    total = sum(num)
    var = []
    for nk in num:
        var.append(nk * (total - nk) / total)
    
    return np.array(var)

def protectedDiv(left, right):
    with np.errstate(divide='ignore',invalid='ignore'):
        y = np.divide(left, right)
        if isinstance(y, np.ndarray):
            y[np.isinf(y)] = 1
            y[np.isnan(y)] = 1
        elif np.isinf(y) or np.isnan(y):
            y = 1
    return y

def protectedPower(left,right):
    with np.errstate(over='ignore',invalid='ignore'):
        y = np.power(left, right)
        if isinstance(y, np.ndarray):
            y[np.isinf(y)] = 1
            y[np.isnan(y)] = 1
        elif np.isinf(y) or np.isnan(y):
            y = 1
    return y

def makeNodeSet():
    nodeSet = []
    leafSet = []
    nodeSet.append(("add",np.add,2))
    nodeSet.append(("sub",np.subtract,2))
    nodeSet.append(("mul",np.multiply,2))
    nodeSet.append(("div",protectedDiv,2))
    # nodeSet.append(("pow",np.power,2))
    nodeSet.append(("sin",np.sin,1))
    nodeSet.append(("cos",np.cos,1))
    nodeSet.append(("tan",np.tan,1))
    nodeSet.append(("log",np.log,1))
    nodeSet.append(("exp",np.exp,1))
    nodeSet.append(("sqrt",np.sqrt,1))

    leafSet.append(("-1",-1,0))
    leafSet.append(("0",0,0))
    leafSet.append(("1",1,0))
    leafSet.append(("0.5",0.5,0))
    leafSet.append(("0.1",0.1,0))
    leafSet.append(("pi",np.pi,0))
    leafSet.append(("e",np.e,0))
    leafSet.append(("x",None,0))

    return nodeSet,leafSet

def randomTreeMake(nodeSet,leafSet,_min,_max):
    """木(個体)の生成

    Args:
        stack (list): 各ノードの木の深さを表す． (ex. stack = [2, 2] -> 子ノード2個のそれぞれの深さを表す)

    Returns:
        tree object: 非終端，終端記号のタプルを要素に持つリスト
    """
    height = random.randint(_min,_max)
    items = []
    stack = [0]
    while len(stack) != 0:
        depth = stack.pop()
        if depth == height:
            node = random.choice(leafSet)
        else:
            node = random.choice(nodeSet)
            stack.extend([depth+1]*node[2])
        items.append(node)  # 格納される順番は行きがけ順
    tree = Tree(items)
    return tree

def get_y(tree,nodeSet,leafSet,x):
    stack = []
    for node in tree[::-1]:
        name,func,arity = node
        if name == "x":
            output = x
        else:
            if arity == 0:
                output = func*np.ones(x.shape[0])
            if arity == 1:
                outout = func(stack.pop())
            if arity == 2:
                output = func(stack.pop(),stack.pop())
        stack.append(output)
    return stack.pop()

def mate(tree1,tree2,limit,limMode):
    save1 = deepcopy(tree1)
    save2 = deepcopy(tree2)
    while True:
        index1 = random.randint(1,len(tree1)-1)
        index2 = random.randint(1,len(tree2)-1)
        slice1 = tree1.searchSubtree(index1)
        slice2 = tree2.searchSubtree(index2)
        tree1[slice1],tree2[slice2] = tree2[slice2],tree1[slice1]
        if limMode == "LENGTH":
            if len(tree1) <= limit and len(tree2) <= limit:
                return tree1,tree2
        elif limMode == "HEIGHT":
            if tree1.height <= limit and tree2.height <= limit:
                return tree1,tree2
        else:
            return tree1,tree2
        tree1,tree2 = deepcopy(save1),deepcopy(save2)  # 交叉して生成された木が最大深さを超えた場合，元の木をそのまま使用．

def mutate(tree,NodeSet,leafSet,limit,limMode):
    save = deepcopy(tree)
    while True:
        index = random.randrange(len(tree))
        slice_ = tree.searchSubtree(index)
        tree[slice_] = randomTreeMake(NodeSet,leafSet,1,2)
        if limMode == "LENGTH":
            if len(tree) <= limit:
                return tree
        elif limMode == "HEIGHT":
            if tree.height <= limit:
                return tree
        else:
            return tree
        tree = deepcopy(save)

def initial_population(N,nodeSet,leafSet,_min,_max):
    """初期集団（個体群）生成関数

    Args:
        N (int): 生成する個体数
        nodeSet (list): 非終端記号
        leafSet (list): 終端記号
        _min (int): 初期個体の最小の木の深さ
        _max (int): 初期個体の最大の木の深さ

    Returns:
        list: treeインスタンス([])
    """
    population = []
    for n in range(N):
        tree = randomTreeMake(nodeSet,leafSet,_min,_max)
        population.append(tree)
    return population

def evaluate(population,nodeSet,leafSet,x,y, xknots, yknots):
    M = 10
    E = 10
    for tree in population:
        if tree.fitness == None:
            _y = get_y(tree,nodeSet,leafSet,x)
            _y2 = get_y(tree,nodeSet, leafSet, xknots)  # スプライン節点における生成関数の値
            # fitness = var * ((_y - y) ** 2)
            fitness = M * np.sum(np.abs(_y2 - yknots)) + np.sum(np.abs(_y - y))
            fitness = M * np.sum(np.abs(_y2 - yknots)) + E * (np.abs(_y[0] - y[0])) + E * (np.abs(_y[-1] - y[-1])) + np.sum(np.abs(_y[1:-1] - y[1:-1]))
            # 各データ点における残差の総和(最後の結果比較で使用する)
            tree.fitness = fitness

def select(population,M):
    new_population = []
    for i in range(len(population)):
        sub_population = random.sample(population,M)
        best = min(sub_population,key=lambda tree:tree.fitness)
        new_population.append(deepcopy(best))
    return new_population

def crossover(population,Pcx):
    random.shuffle(population)
    for i in range(int(len(population)/2)):
        if random.random() < Pcx:
            childA = population[2*i]
            childB = population[2*i+1]
            childA,childB = mate(childA,childB,9,"HEIGHT")
            population[2*i] = deepcopy(childA)
            population[2*i+1] = deepcopy(childB)
            population[2*i].fitness = None
            population[2*i+1].fitness = None
    return population

def mutation(population,Pmut,nodeSet,leafSet):
    random.shuffle(population)
    for i in range(len(population)):
        if random.random() < Pmut:
            child = population[i]
            child = mutate(child,nodeSet,leafSet,9,"HEIGHT")
            population[i] = deepcopy(child)
            population[i].fitness = None

    return population

def printLog(population,g):
    key=lambda tree:tree.fitness
    worst = max(population,key=key)
    best = min(population,key=key)
    ave = sum([tree.fitness for tree in population])/len(population)
    key=lambda tree:len(tree)
    max_node = len(max(population,key=key))
    min_node = len(min(population,key=key))
    ave_node = sum([len(tree) for tree in population])/len(population)
    print("===========",g,"genelation===========")
    print("worst fitness : ",worst.fitness,"  max nodes : ",max_node)
    print("ave   fitness : ",ave          ,"  ave nodes : ",ave_node)
    print("best  fitness : ",best.fitness, "  min nodes : ",min_node)


def calc_result(best, nodeSet, leafSet, x, y):
    _y = get_y(best,nodeSet,leafSet,x)
    norm_y = np.sum((stats.norm.pdf(x, loc=54, scale=21.2) - y) ** 2)
    result = np.sum((_y - y) ** 2)
    print("best of GP residual: {}".format(result))
    print("norm residual: {}".format(norm_y))


def makeGraph(tree,x,y,nodeSet,leafSet, xknots, yknots):
    _y = get_y(tree,nodeSet,leafSet,x)
    plt.plot(x,y,label="TRUE")
    plt.plot(x,_y,label="GP")
    plt.plot(x, stats.norm.pdf(x, loc=54, scale=21.2), label='NORM')
    plt.scatter(x, y, label="dataset")
    plt.scatter(xknots, yknots, label='spline node')
    plt.xlabel('Score')
    plt.ylabel('Probability density')
    # plt.grid(color='gray', ls='dotted', lw=0.5)
    plt.legend(loc="upper left")
    plt.savefig("test.pdf")
    plt.show()

def main():
    """main function

    Args:
        N: 個体数
        M: トーナメント選択数
        maxG: 最大世代数
        Pcx: 交叉率
        Pmut: 突然変異率
        nodeSet: 非終端記号
        leafSet: 終端記号
        pupulation: 個体群
    """
    random.seed(1032)
    N = 500
    M = 8
    maxG = 200
    Pcx = 0.5
    Pmut = 0.1
    nodeSet,leafSet = makeNodeSet()
    x, y, xknots, yknots, alldata = generate_dataset(input_data.dataset)

    population = initial_population(N,nodeSet,leafSet, MIN_DEPTH, MAX_DEPTH)
    g = 0
    while True:
        evaluate(population,nodeSet,leafSet,x,y,xknots,yknots)
        printLog(population,g)
        if g == maxG:
            break
        population = select(population,M)
        population = crossover(population,Pcx)
        population = mutation(population,Pmut,nodeSet,leafSet)
        g+=1
    best = min(population,key=lambda tree:tree.fitness)
    print(best)
    makeGraph(best,x,y,nodeSet,leafSet, xknots, yknots)
    calc_result(best, nodeSet, leafSet, x, y)

main()

# for i in range(1016, 1050):
#     random.seed(i)
#     main()
#     print("seed is {}".format(i))
