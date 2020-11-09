import numpy as np
import multiprocessing
from sko.GA import GA
from sko.PSO import PSO
from sko.DE import DE
import matplotlib.pyplot as plt
import random
import math
import random
from copy import deepcopy
from math import sin, cos, log


def P_min(f):
    other_mins = []
    de = DE(func=f, n_dim=DIM, lb=LB, ub=UB, max_iter=ITER // 2, size_pop=POP)
    ga = GA(func=f, n_dim=DIM, lb=LB, ub=UB, max_iter=ITER, size_pop=POP, precision=1e-200)
    pso = PSO(func=f, dim=DIM, pop=POP, max_iter=ITER, lb=LB, ub=UB)
    _, e_de = de.run()
    e_de = e_de[0]
    _, e_ga = ga.run()
    e_ga = e_ga[0]
    pso.run()
    e_pso = pso.gbest_y
    time_de, time_ga, time_pso = 0, 0, 0
    cnt = 0
    while True:
        de_x, temp_de = de.run()
        temp_de = temp_de[0]
        time_de = time_de + 1 if e_de - temp_de < CALC_PRECISION else 0
        e_de = temp_de
        ga_x, temp_ga = ga.run()
        temp_ga = temp_ga[0]
        time_ga = time_ga + 1 if e_ga - temp_ga < CALC_PRECISION else 0
        e_ga = temp_ga
        pso.run()
        time_pso = time_pso + 1 if e_pso - pso.gbest_y < CALC_PRECISION else 0
        e_pso = pso.gbest_y
        res = sorted([[e_de, time_de, de_x], [e_ga, time_ga, ga_x], [e_pso, time_pso, pso.gbest_x]], key = lambda x:x[0])
        if res[1][0] - res[0][0] < CALC_PRECISION and res[0][1] > STOP_ITER_TIME:
            return (res[0][0],res[0][2])
        if res[0][1] > STOP_ITER_TIME and res[1][1] > STOP_ITER_TIME and res[2][1] > STOP_ITER_TIME:
            other_mins.append((res[0][0],res[0][2]))
            if len(other_mins) == 10:
                return min(other_mins, key = lambda x:x[0])
            res = []
            de = DE(func=f, n_dim=DIM, lb=LB, ub=UB, max_iter=ITER // 2, size_pop=POP)
            ga = GA(func=f, n_dim=DIM, lb=LB, ub=UB, max_iter=ITER, size_pop=POP, precision=1e-200)
            pso = PSO(func=f, dim=DIM, pop=POP, max_iter=ITER, lb=LB, ub=UB)
            de_x, e_de = de.run()
            e_de = e_de[0]
            ga_x, e_ga = ga.run()
            e_ga = e_ga[0]
            pso.run()
            e_pso = pso.gbest_y
            time_de, time_ga, time_pso = 0, 0, 0


def reg_time(args):
    #假设TEST_TIME为100
    #总运行次数小于T1(比如200)时，运行次数就是运行次数
    #总运行次数大于T1(比如200)时，计算运行次数乘以TEST_TIME/实际运行次数与T2的调和平均数
    #这里T1代表了实际最大运算次数，T2代表了计算结果的最大重算次数
    kind, f, p_min = args
    print(".", end="")
    res_ary = [1e+500]
    if kind == "de":
        de = DE(func=f, n_dim=DIM, lb=LB, ub=UB, max_iter=ITER // 2, size_pop=POP)
        time_de = 0
        while res_ary[-1] > p_min + MIN_PRECISION:
            if len(res_ary)>STOP_ITER_TIME and res_ary[-STOP_ITER_TIME]-res_ary[-1] < CALC_PRECISION:
                return (False, time_de-STOP_ITER_TIME)
            _, temp_de = de.run()
            res_ary.append(temp_de[0])
            time_de += 1
        return (True, time_de)
    if kind == "ga":
        ga = GA(func=f, n_dim=DIM, lb=LB, ub=UB, max_iter=ITER, size_pop=POP, precision=1e-200)
        time_ga = 0
        while res_ary[-1] > p_min+MIN_PRECISION:
            if len(res_ary)>STOP_ITER_TIME and res_ary[-STOP_ITER_TIME]-res_ary[-1] < CALC_PRECISION:
                return (False, time_ga-STOP_ITER_TIME)
            _, temp_ga = ga.run()
            res_ary.append(temp_ga[0])
            time_ga += 1
        return (True, time_ga)
    if kind == "pso":
        pso = PSO(func=f, dim=DIM, pop=POP, max_iter=ITER, lb=LB, ub=UB)
        time_pso = 0
        while res_ary[-1] > p_min+MIN_PRECISION:
            if len(res_ary)>STOP_ITER_TIME and res_ary[-STOP_ITER_TIME]-res_ary[-1] < CALC_PRECISION:
                return (False, time_pso-STOP_ITER_TIME)
            pso.run()
            res_ary.append(pso.gbest_y)
            time_pso += 1
        return (True, time_pso)
def de_BD(f):
    p_min, p_min_x = P_min(f)
    print(p_min_x)
    for i in range(DIM):
        if (MIN_PRECISION - 1 < p_min_x[i] < 1 - MIN_PRECISION):
            break
    else:
        # print("min_x at edge.")
        return 1e7
    if f([1 for _ in range(DIM)]) - p_min < CALC_PRECISION:
        # print("const func.")
        return 1e7
    all_cnt = 0
    right_cnt = 0
    times = 0
    while True:
        s = map(reg_time, [("de", f, p_min)] * TEST_TIME)
        for i in s:
            if i[0]:
                right_cnt += 1
            times += i[1]
            all_cnt+=1
            if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
                break
        if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
            break
    de_score = times/all_cnt/(right_cnt/all_cnt+ MIN_TEST_ACC)
    # print("。")
    all_cnt = 0
    right_cnt = 0
    times = 0
    while True:
        s = map(reg_time, [("ga", f, p_min)] * TEST_TIME)
        for i in s:
            if i[0]:
                right_cnt += 1
            times += i[1]
            all_cnt+=1
            if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
                break
        if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
            break
    ga_score = times/all_cnt/(right_cnt/all_cnt+ MIN_TEST_ACC)
    # print("。")
    all_cnt = 0
    right_cnt = 0
    times = 0
    while True:
        s = map(reg_time, [("pso", f, p_min)] * TEST_TIME)
        for i in s:
            if i[0]:
                right_cnt += 1
            times += i[1]
            all_cnt+=1
            if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
                break
        if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
            break
    pso_score = times/all_cnt/(right_cnt/all_cnt+ MIN_TEST_ACC)
    # print("。")
    print("avg", de_score, ga_score, pso_score)
    print("score", -de_score / pso_score if pso_score > ga_score else -de_score / ga_score)
    return -de_score / pso_score if pso_score > ga_score else -de_score / ga_score
def ga_BD(f):
    p_min, p_min_x = P_min(f)
    print(p_min_x)
    for i in range(DIM):
        if (MIN_PRECISION - 1 < p_min_x[i] < 1 - MIN_PRECISION):
            break
    else:
        # print("min_x at edge.")
        return 1e7
    if f([1 for _ in range(DIM)]) - p_min < CALC_PRECISION:
        # print("const func.")
        return 1e7
    all_cnt = 0
    right_cnt = 0
    times = 0
    while True:
        s = map(reg_time, [("de", f, p_min)] * TEST_TIME)
        for i in s:
            if i[0]:
                right_cnt += 1
            times += i[1]
            all_cnt+=1
            if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
                break
        if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
            break
    de_score = times/all_cnt/(right_cnt/all_cnt+ MIN_TEST_ACC)
    # print("。")
    all_cnt = 0
    right_cnt = 0
    times = 0
    while True:
        s = map(reg_time, [("ga", f, p_min)] * TEST_TIME)
        for i in s:
            if i[0]:
                right_cnt += 1
            times += i[1]
            all_cnt+=1
            if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
                break
        if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
            break
    ga_score = times/all_cnt/(right_cnt/all_cnt+ MIN_TEST_ACC)
    # print("。")
    all_cnt = 0
    right_cnt = 0
    times = 0
    while True:
        s = map(reg_time, [("pso", f, p_min)] * TEST_TIME)
        for i in s:
            if i[0]:
                right_cnt += 1
            times += i[1]
            all_cnt+=1
            if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
                break
        if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
            break
    pso_score = times/all_cnt/(right_cnt/all_cnt+ MIN_TEST_ACC)
    # print("。")
    print("avg", de_score, ga_score, pso_score)
    print("score", -ga_score / pso_score if pso_score > de_score else -ga_score / de_score)
    return -ga_score / pso_score if pso_score > de_score else -ga_score / de_score

def pso_BD(f):
    p_min, p_min_x = P_min(f)
    # print(p_min)
    print(p_min_x)
    for i in range(DIM):
        if (MIN_PRECISION - 1 < p_min_x[i] < 1 - MIN_PRECISION):
            break
    else:
        # print("min_x at edge.")
        return 1e7
    if f([1 for _ in range(DIM)]) - p_min < CALC_PRECISION:
        # print("const func.")
        return 1e7
    all_cnt = 0
    right_cnt = 0
    times = 0
    while True:
        s = map(reg_time, [("de", f, p_min)] * TEST_TIME)
        for i in s:
            if i[0]:
                right_cnt += 1
            times += i[1]
            all_cnt+=1
            if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
                break
        if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
            break
    de_score = times/all_cnt/(right_cnt/all_cnt+ MIN_TEST_ACC)
    # print("。")
    all_cnt = 0
    right_cnt = 0
    times = 0
    while True:
        s = map(reg_time, [("ga", f, p_min)] * TEST_TIME)
        for i in s:
            if i[0]:
                right_cnt += 1
            times += i[1]
            all_cnt+=1
            if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
                break
        if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
            break
    ga_score = times/all_cnt/(right_cnt/all_cnt+ MIN_TEST_ACC)
    # print("。")
    all_cnt = 0
    right_cnt = 0
    times = 0
    while True:
        s = map(reg_time, [("pso", f, p_min)] * TEST_TIME)
        for i in s:
            if i[0]:
                right_cnt += 1
            times += i[1]
            all_cnt+=1
            if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
                break
        if TEST_TIME <= right_cnt or MAX_TEST_TIME <= all_cnt:
            break
    pso_score = times/all_cnt/(right_cnt/all_cnt+ MIN_TEST_ACC)
    # print("。")
    print("avg", de_score, ga_score, pso_score)
    print("score", -pso_score / ga_score if ga_score > de_score else -pso_score / de_score)
    return -pso_score / ga_score if ga_score > de_score else -pso_score / de_score


class TreeNode:
    # vars: weight, opkind, op, dim, children,is_root, has_weight, offsprings
    def __init__(self, root=None, depth=0):
        self.depth = depth
        if root is None:
            self.length = 1
            self.is_root = True
            self.has_weight = False
        else:
            self.is_root = False
            if random.random() < TreeNode.rootspecialp:
                pass  # special_func
        if random.random() < TreeNode.funcp - depth * TreeNode.suppressing_coefficient:
            option = random.choice(calcs)
            self.op = option[0]
            self.dim = option[1]
            self.children = [TreeNode(self, depth + 1) for _ in range(self.dim)]
        elif random.random() < TreeNode.specialp:
            pass  # special_func
        elif random.random() < TreeNode.constp:
            self.op = f"({-math.log(1 / random.random() - 1)})"
            self.dim = 0
        else:
            self.op = f"var[{random.randint(0, DIM - 1)}]"
            self.dim = 0

    def init_calc(self):
        self.raw_calc = eval("lambda var:" + str(self))
        s = str(self)
        for i in range(DIM):
            v1 = [1 if j == i else 0 for j in range(DIM)]
            v2 = [-1 if j == i else 0 for j in range(DIM)]
            s += f"-({(self.raw_calc(v1) - self.raw_calc(v2)) / 2}*var[{i}])"
        self.__str__ = lambda: s
        self.calc = eval("lambda var:" + s)

    def __str__(self):
        s = self.op
        for i in range(self.dim):
            s = s.replace(f"x{i}", str(self.children[i]))
        return s

    def fitness(self):  # 适应度
        # print(self)
        self.init_calc()
        self.score = Environment.func(self.calc)
        return self.score

    def variation(self, p=0, root=None, depth=0):
        if root is None:
            root = self
        if random.random() < p:
            return TreeNode(root, depth)
        elif root == self:
            t = deepcopy(self)
        else:
            t = self
        if self.dim > 0:
            t.children = [i.variation(p + 0.1, root, depth + 1) for i in t.children]
        return t

    def intersect(self, tree, p=0, root=None, depth=0):
        if root is None:
            root = self
        if random.random() < p:
            return tree.subtree(depth)
        elif root == self:
            t = deepcopy(self)
        else:
            t = self
        if self.dim > 0:
            t.children = [i.variation(p + 0.1, root, depth + 1) for i in t.children]
        return t

    def subtree(self, depth):
        kids = [self]
        temp_kids = []
        for i in range(depth):
            for j in kids:
                if j.dim > 0:
                    temp_kids += j.children
            kids = temp_kids
            temp_kids = []
        return random.choice(kids)

class Environment:
    def __init__(self, init_ary=None):
        self.best_t = None
        self.best_score = 10000000
        self.population = [0 for _ in range(Environment.size*5)]
        for i in range(Environment.size*5):
            self.population[i] = TreeNode()
        pool = multiprocessing.Pool(processes=4)
        scores = pool.map(TreeNode.fitness, self.population)
        pool.terminate()
        print(scores)
        for i in range(Environment.size*5):
            self.population[i].score = scores[i]
        self.population.sort(key=lambda x: x.score)
        self.population = self.population[:Environment.size]

    def gen(self):
        temp_population = []
        temp_population.append(self.population[0])
        temp_population.append(self.population[0].intersect(self.population[1]))
        temp_population.append(self.population[0].intersect(self.population[1]))
        temp_population.append(self.population[0].intersect(self.population[1]))
        temp_population.append(self.population[0].intersect(self.population[2]))
        temp_population.append(self.population[0].intersect(self.population[2]))
        temp_population.append(self.population[0].intersect(self.population[3]))
        temp_population.append(self.population[0].variation())
        temp_population.append(self.population[0].variation())
        temp_population.append(self.population[1])
        temp_population.append(self.population[1].intersect(self.population[0]))
        temp_population.append(self.population[1].intersect(self.population[0]))
        temp_population.append(self.population[1].intersect(self.population[2]))
        temp_population.append(self.population[1].variation())
        temp_population.append(self.population[2])
        temp_population.append(self.population[2].intersect(self.population[0]))
        temp_population.append(self.population[2].intersect(self.population[1]))
        temp_population.append(self.population[2].variation())
        temp_population.append(self.population[3])
        temp_population.append(self.population[3].intersect(self.population[0]))
        temp_population.append(self.population[3].variation())
        temp_population.append(self.population[4])
        temp_population.append(self.population[4].intersect(self.population[0]))
        for i in range(5, 20):
            temp_population.append(self.population[i].intersect(self.population[random.randint(0, 29)]))
        for _ in range(Environment.size - 38):
            temp_population.append(TreeNode())
        self.population = temp_population
        pool = multiprocessing.Pool(processes=4)
        scores = pool.map(TreeNode.fitness, [self.population[i] for i in range(Environment.size)])
        pool.terminate()
        for i in range(Environment.size):
            self.population[i].score = scores[i]
        self.population.sort(key=lambda x: x.score)

KIND_FUNC = 1
KIND_CONST = 2
KIND_VAR = 3
res_list, de_BE_cnt = [], 0
DIM = 2
ITER, POP, LB, UB, STOP_ITER_TIME, CALC_PRECISION, MIN_PRECISION, MAX_CALC_MEAN_TIME = 2, 50, [-1]*DIM, [1]*DIM, 200, 1e-11, 1e-10, 10
TEST_TIME, MAX_TEST_TIME, MIN_TEST_ACC = 10, 200, 0.005
file = open("output.txt", 'w')
calcs = [("(x0+x1)", 2), ("(x0*x1)", 2), ("sin(x0)", 1), ("cos(x0)", 1), ("log(abs(x0)+1)", 1)]
Environment.size = 40
Environment.func = pso_BD
TreeNode.rootspecialp = 0
TreeNode.specialp = 0
TreeNode.funcp = 1
TreeNode.suppressing_coefficient = 0.1
TreeNode.constp = 0.2

if __name__ == '__main__':
    print("gen", 1)
    env = Environment()
    print(list(map(str, env.population)))
    print(env.population[0].score)
    for i in range(29):
        print("gen", i+2)
        env.gen()
        print(list(map(str, env.population)))
        print(env.population[0].score)