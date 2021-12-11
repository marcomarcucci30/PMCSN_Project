from utils.rngs import random, selectStream, plantSeeds
from utils.rvgs import Exponential, TruncatedNormal

nodes = 4  # n nodi
arrival_rate = 1.0
seed = 123456789  # TODO: Controlla il seed migliore o forse era il multiplier?
START = 0.0
STOP = 1440.0  # Minutes
INFINITY = STOP * 100.0
p0 = 0.8
TICKET_QUEUE = 0
ARCADE1 = 1
ARCADE2 = 2
ARCADE3 = 3
p_size = 0.8


class Track:
    node = 0.0  # time integrated number in the node
    queue = 0.0  # time integrated number in the queue
    service = 0.0  # time integrated number in service


class Time:
    current = None  # current time
    next = None  # next (most imminent) event time


time = Time()


class StatusNode:
    id = None
    arrival = None  # next arrival time
    completion = None  # next completion time
    last = 0.0  # last arrival time
    index = 0  # jobs departed
    number = 0  # jobs in node
    stat = Track()  # Track stats

    def __init__(self, id):
        self.id = id


def set_arrival_rate(x):
    global arrival_rate
    arrival_rate = x


def get_arrival_rate():
    return arrival_rate


def get_arrival(y):
    # ---------------------------------------------
    # * generate the next arrival time from an Exponential distribution.
    # * --------------------------------------------

    selectStream(0)
    return Exponential(y)


def select_node(from_tkt_queue):
    selectStream(5)  # TODO: scegliere gli streams sequenziali o no?
    if from_tkt_queue:
        r = random()
        if r <= 1 / (nodes - 1):
            return ARCADE1
        elif r <= 2 / (nodes - 1):
            return ARCADE2
        else:
            return ARCADE3
    r = random()
    if r <= p0:
        return TICKET_QUEUE
    else:
        r = random()
        if r <= 1 / (nodes - 1):
            return ARCADE1
        elif r <= 2 / (nodes - 1):
            return ARCADE2
        else:
            return ARCADE3


def minimum(a, b):
    if a is None and b is not None:
        return b
    elif b is None and a is not None:
        return a
    elif a is None and b is None:
        return None
    elif a < b:
        return a
    else:
        return b


def next_event():
    pos = 0
    for i in range(1, len(node_list) - 1):
        min_local = minimum(node_list[i].arrival, node_list[i].completion)
        min_local_next = minimum(node_list[i + 1].arrival, node_list[i + 1].completion)
        min_local = minimum(min_local, min_local_next)
        if min_local == minimum(min_local, min_local_next):
            if min_local == minimum(min_local, n)
            pos = i
        else:
            pos = i + 1
    return pos


def get_service(id):
    # --------------------------------------------
    # * generate the next service time
    # * --------------------------------------------
    # */
    if id == TICKET_QUEUE:
        selectStream(6)
        r = random()
        if r <= p_size:  # green pass
            selectStream(id + 10)
            return TruncatedNormal(2, 1.5, 1, 3)  # green pass
        else:
            selectStream(id + 15)
            return TruncatedNormal(10, 1.5, 8, 12)  # covid test
    else:
        selectStream(id + 10)
        return TruncatedNormal(15, 3, 3, 25)  # arcade game time


if __name__ == '__main__':
    # settings
    node_list = [StatusNode(i) for i in range(nodes + 1)]  # in 0 global stats
    plantSeeds(seed)

    time.current = START
    arrival = START  # global temp var for getArrival function     [minutes]

    # initialization of the first arrival event
    arrival += get_arrival(arrival_rate)
    node = node_list[select_node(False) + 1]
    node.arrival = arrival
    min_arrival = arrival

    while (min_arrival < STOP) or (node_list[0].number > 0):
        node_to_process = node_list[next_event()]  # node with minimum arrival or completion time
        time.next = minimum(node_to_process.arrival, node_to_process.completion)
        # Aggiornamento delle aree basate sul giro prima
        for i in range(0, len(node_list)):
            if node_list[i].number > 0:
                node_list[i].stat.node = (time.next - time.current) * node_list[i].number
                node_list[i].stat.queue = (time.next - time.current) * (node_list[i].number - 1)
                node_list[i].stat.service = (time.next - time.current)

        time.current = time.next  # advance the clock

        if time.current == node_to_process.arrival:
            node_to_process.number += 1
            node_list[0].number += 1  # update system stat
            arrival += get_arrival(arrival_rate)
            node = node_list[select_node(False) + 1]

            # Controllo che l'arrivo sul nodo i-esimo sia valido. In caso negativo
            # imposto come ultimo arrivo del nodo i-esimo l'arrivo precedentemente
            # considerato
            if arrival > STOP:
                node.last = node.arrival
                # update node and system last arrival time
                if node_list[0].last < node.last:
                    node_list[0].last = node.last
                node.arrival = INFINITY
            else:
                node.arrival = arrival

            if node_to_process.number == 1:
                node_to_process.completion = time.current + get_service(node_to_process.id)
        else:
            node_to_process.index += 1  # node stats update
            node_to_process.number -= 1
            if node_to_process.id != TICKET_QUEUE:  # system stats update
                node_list[0].index += 1
                node_list[0].number -= 1
            if node_to_process.number > 0:
                node_to_process.completion = time.current + get_service(node_to_process.id)
            else:
                node_to_process.completion = INFINITY

            if node_to_process.id == TICKET_QUEUE:  # a completion on TICKET_QUEUE trigger an arrival on ARCADE_i
                arcade_node = node_list[select_node(True) + 1]  # on first global stats
                arcade_node.number += 1  # system stats don't updated
                if arcade_node.number == 1:
                    arcade_node.completion = time.current + get_service(arcade_node.id)

        arrival_list = [n.arrival for n in node_list]
        min_arrival = min(arrival_list)

    print("\nfor {0} jobs".format(node_list[0].index))
    print("   average interarrival time = {0:6.2f}".format(node_list[0].last / node_list[0].index))
    print("   average wait ............ = {0:6.2f}".format(node_list[0].stat.node / node_list[0].index))
    print("   average delay ........... = {0:6.2f}".format(node_list[0].stat.queue / node_list[0].index))
    print("   average # in the node ... = {0:6.2f}".format(node_list[0].stat.node / time.current))
    print("   average # in the queue .. = {0:6.2f}".format(node_list[0].stat.queue / time.current))
    print("   utilization ............. = {0:6.2f}".format(node_list[0].stat.service / time.current))
