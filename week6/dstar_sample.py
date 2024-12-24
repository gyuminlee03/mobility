##########################################################
##
## 필요한 라이브러리 import
## 전역 변수 선언
##
##########################################################
import math
from sys import maxsize
import matplotlib.pyplot as plt

show_animation = True



##########################################################
##
## state 클래스:
## 각 격자의 state(위치, 상태, cost 등)를 나타냄
## 상태표현:
##     .: new
##     #: obstacle
##     e: parent of current state
##     *: closed state
##     s: current state
##
##########################################################
class State:
    #########################################
    ## 초기화 함수: 입력된 위치의 node를 초기화(new state)
    #########################################
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."
        self.t = "new"  # tag for state 
        self.h = 0  # cost
        self.k = 0  # cost 

    #########################################
    ## cost 계산: 현재의 node와 입력된 node 간의 cost 계산, 둘 중 하나라도 장애물이면 inf
    #########################################
    def cost(self, state):
        if self.state == "#" or state.state == "#":
            return maxsize

        return math.sqrt(math.pow((self.x - state.x), 2) +
                         math.pow((self.y - state.y), 2))


    #########################################
    ## state 변경 
    #########################################
    def set_state(self, state):
        if state not in ["s", ".", "#", "e", "*"]:
            return
        self.state = state

##########################################################
##
## map 클래스:
## 주행 환경을 격자 지도로 표현
##
##########################################################
class Map:

    #########################################
    ## 초기화 함수
    #########################################
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    #########################################
    ## 지도 초기화: 정해진 크기의 격자 지도를 초기화, 모든 node가 new state
    #########################################
    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    #########################################
    ## 입력된 node의 이웃(상하좌우 대각선 1칸) node 반환
    #########################################
    def get_neighbors(self, node):
        node_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if node.x + i < 0 or node.x + i >= self.row:
                    continue
                if node.y + j < 0 or node.y + j >= self.col:
                    continue
                node_list.append(self.map[node.x + i][node.y + j])
        return node_list

    #########################################
    ## 장애물 node 설정
    #########################################
    def set_obstacle(self, point_list):
        for x, y in point_list:
            if x < 0 or x >= self.row or y < 0 or y >= self.col:
                continue
            self.map[x][y].set_state("#")










##########################################################
##
## Dstar 클래스:
## D star 알고리즘 구동을 위한 함수 포함
## OPEN: 아직 최적 cost 계산이 되지 않아 방문해야하는 node
## CLSOE: 최적 cost 계산이 완료된 node
## h = cost
#x fh rjcurksms cost rk ej skwdmfEo yfh rjcurksmsrjtqhek ej skdmsrjdla -> rmEo yfmf djqepdlxm -> insert() gownrl!
##########################################################


class Dstar:




    #########################################
    ## 초기화 함수
    #########################################
    def __init__(self, maps):
        self.map = maps # map bataogi
        self.open_list = set() # open list input!





    #########################################
    ## D* 트리 확장 및 업데이트(to-do)
    #########################################
    def process_state(self):
        #=============
        # open_list에서 가장 작은 cost의 노드와 그 cost를 가져옴
        #start = 0, namuji = inf or NON

        # 만약 open_list가 비어있다면 함수 종료
        
        # 위에서 가져온 node를 open_list에서 제거
        # neigbor node search
        # -> current -> next node go if cost + now node cost(nu juk)

        # 초기 cost 계산 (정적인 지도 가정)

        # 환경 변화로 인해 cost가 달라진 경우 트리 업데이트
        #
        #=============
        #open_list low cost, cost get
        x = self.min_state()
        k_old = self.get_kmin()

        # if open cost is empty = reutrn 0
        if x is None:
            return -1

        # node get, open list eleminate
        self.remove(x)

        
        #initial cost calculate (error when, -> can't fix code ////)
     #   if k_old == x.h:
      #      for y in self.map.get_neighbors(x):
       #         if(y.t == "new") or (y.parent != x and y.h > x.h + x.cost(y)): # if tag is new, or parent is not x, y cost > x cost -> x cost
         #           y.parent = x
        #            self.insert(y, x.h + x.cost(y))



    # fixed code!
        if k_old == x.h:
         for y in self.map.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        if k_old != x.h:
            for y in self.map.get_neighbors(x):
                if(y.t == "new") or (y.parent != x and y.h > x.h + x.cost(y)): # if tag is new, or parent is not x, y cost > x cost -> x cost
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and x.h > y.h + x.cost(y) \
                            and y.t == "close" and y.h > k_old:
                        self.insert(y,y.h)



        #modify func use
        

        #self.modify_cost(x.cost(y))
        #self.modify(y)
        
        #AddNewObstacle(self.map)


        




        return self.get_kmin()




    #########################################
    ## open_list에서 최소 k를 가지고 있는 노드 반환
    #########################################
    def min_state(self):
        if not self.open_list:
            return None
        #min_state = min(self.open_list, key=lambda x: x.k)
        min_state = min(self.open_list, key=lambda x: x.k) # -> X = open list dml element 
        # min cost element come here! -> min_state(good~~~)

        return min_state




    #########################################
    ## open_list에서 최소 k를 반환
    #########################################
    def get_kmin(self):
        if not self.open_list:
            return 100
        k_min = min([x.k for x in self.open_list])
        return k_min




    #########################################
    ## open_list에 새로운 node를 삽입
    #########################################
    def insert(self, state, h_new): # state = node
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)





    #########################################
    ## open_list에서 입력 node를 삭제
    #########################################
    def remove(self, state):
        if state.t == "open":
            state.t = "close"
        self.open_list.remove(state)





    #########################################
    ## 환경 변화 등으로 인한 트리 수정
    #########################################
    def modify(self, state):
        self.modify_cost(state)
        while True:
            k_min = self.process_state()
            if k_min >= state.h:
                break




    #########################################
    ## 환경 변화 등으로 인한 cost 수정
    #########################################
    def modify_cost(self, x):
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))
            # parent = error, wall changed
            # parent -> "infinite" fh change 
            #that cost giban, we can update it 












    #########################################
    ## 알고리즘 구동
    #########################################
    def run(self, start, end):

        rx = [] # 최종 경로
        ry = []

        self.insert(end, 0.0) # 최종점 삽입  wmr, goal jijum -> start  (g -> s) ((yuk)Reverse search ) gogogo!!!!

        # 초기 지도에서 최적 cost 생성
        while True:
            self.process_state()
            if start.t == "close":
                break

        start.set_state("s")
        s = start
        s = s.parent
        s.set_state("e")
        current = start

        # 초기 지도에는 없었던 새로운 장애물 생성
        AddNewObstacle(self.map)   # suddenly error , jang ae mul balseng!!!!!!!!!!!!!!!!

        # 새로운 장애물에 대한 업데이트 및 새로운 경로 생성
        while current != end:
            current.set_state("*")
            rx.append(current.x) # path save
            ry.append(current.y) # my path save \
            if show_animation:
                plt.plot(rx, ry, "-r")
                plt.pause(0.01)
                # no problem, il te


                #if eerror!!!!!!!!!!!!!!!!!
            if current.parent.state == "#": # 앞에서 계산된 cost를 기반으로 경로를 생성하다가 새로운 장애물을 발견하면 cost를 업데이트
                self.modify(current)
                continue 
            current = current.parent
        current.set_state("e")

        return rx, ry





#########################################
## 새로운 장애물 생성
#########################################
def AddNewObstacle(map:Map):
    ox, oy = [], []
    for i in range(5, 21):
        ox.append(i)
        oy.append(40)
    map.set_obstacle([(i, j) for i, j in zip(ox, oy)])
    if show_animation:
        plt.pause(0.001)
        plt.plot(ox, oy, ".g")





#########################################
## 메인 함수
#########################################
def main():

    # 지도 및 초기 장애물 설정
    m = Map(100, 100)
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10)
    for i in range(-10, 60):
        ox.append(60)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60)
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40)
        oy.append(60 - i)
    m.set_obstacle([(i, j) for i, j in zip(ox, oy)])

    # 시작, 끝 점 설정
    start = [10, 10]
    goal = [50, 50]
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(start[0], start[1], "og")
        plt.plot(goal[0], goal[1], "xb")
        plt.axis("equal")

    # 시작, 끝 node를 가져옴
    start = m.map[start[0]][start[1]]
    end = m.map[goal[0]][goal[1]]

    # Dstar 알고리즘 수행
    dstar = Dstar(m)
    rx, ry = dstar.run(start, end)

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()