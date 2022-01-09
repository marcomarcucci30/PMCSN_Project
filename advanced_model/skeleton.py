from utils.rngs import random, selectStream, plantSeeds
from utils.rvgs import Exponential, TruncatedNormal
from base_model.skeleton import select_node_arrival, select_node_random, select_node_ticket, \
    select_node_arcades, select_node_stream

nodes = 6  # n nodi
arrival_time = 0.0
arrival_time_morning = 14.0
arrival_time_afternoon = 14.0
arrival_time_evening = 14.0
arrival_time_night = 14.0

seed = 1234567891
START = 8.0 * 60
STOP = 15 * 12 * 28 * 1440.0  # Minutes
INFINITY = STOP * 100.0
p_ticket_queue = 0.8
TICKET_QUEUE = 1
p_size = 0.6
p_premium = 0.36
p_positive = 0.05


class Time:
    current = None  # current time
    next = None  # next (most imminent) event time


time = Time()


class Track:
    node = 0.0  # time integrated number in the node
    queue = 0.0  # time integrated number in the queue
    service = 0.0  # time integrated number in service
    index = 0.0  # jobs departed
    number = 0.0  # jobs in node
    last = 0.0

    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0
        self.index = 0.0  # jobs departed
        self.number = 0.0  # jobs in node
        self.last = 0.0


class StatusNode:
    id = None
    arrival = None  # next arrival time
    completion = None  # next completion time
    priority_arrival = False
    priority_completion = False
    last = 0.0  # last arrival time
    more_p_stat = None
    less_p_stat = None

    def __init__(self, id_node):
        self.id = id_node
        self.more_p_stat = Track()  # Track_more_p stats
        self.less_p_stat = Track()  # Track_less_p stats


class SystemTrack:
    node = 0.0  # time integrated number in the node
    queue = 0.0  # time integrated number in the queue
    service = 0.0  # time integrated number in service

    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0


class SystemNode:
    id = None
    arrival = None  # next arrival time
    completion = None  # next completion time
    last = 0.0  # last arrival time
    index = 0.0  # jobs departed
    number = 0.0  # jobs in node
    stat = None

    def __init__(self, id_node):
        self.id = id_node
        self.stat = SystemTrack()  # Track stats


def set_arrival_time(x):
    global arrival_time
    arrival_time = x


def get_arrival_time():
    return arrival_time


arr_est = 0


def select_node(from_tkt_queue):
    selectStream(select_node_stream)
    if from_tkt_queue:
        r = random()
        for i in range(1, nodes):
            if r <= i / (nodes - 1):
                return i + 1
    # Caso arrivo dall'esterno

    r = random()
    if r <= p_ticket_queue:
        global arr_est
        arr_est += 1
        return TICKET_QUEUE
    else:
        r = random()
        for i in range(1, nodes):
            if r <= i / (nodes - 1):
                return i + 1


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


def maximum(a, b):
    if a is None and b is not None:
        return b
    elif b is None and a is not None:
        return a
    elif a is None and b is None:
        return None
    elif a > b:
        return a
    else:
        return b


def next_event():
    time_event = []
    for i in range(1, len(node_list)):
        time_event.append(node_list[i].arrival)
        time_event.append(node_list[i].completion)

    time_event = sorted(time_event, key=lambda x: (x is None, x))

    for i in range(1, len(time_event)):
        if time_event[0] == node_list[i].arrival or time_event[0] == node_list[i].completion:
            return i


def get_arrival(y):
    # ---------------------------------------------
    # * generate the next arrival time from an Exponential distribution.
    # * --------------------------------------------

    selectStream(select_node_arrival)
    return Exponential(y)


def get_service(id_node):
    # --------------------------------------------
    # * generate the next service time
    # * --------------------------------------------
    # */
    if id_node == TICKET_QUEUE:
        if node_list[id_node].priority_completion is True:  # green pass
            selectStream(id_node + select_node_ticket)
            service = TruncatedNormal(2, 1.5, 1, 3)  # green pass
            return service
        else:
            selectStream(id_node + select_node_ticket)
            service = TruncatedNormal(10, 1.5, 8, 12)  # covid test
            return service
    else:
        selectStream(id_node + select_node_arcades)
        service = TruncatedNormal(15, 3, 3, 25)  # arcade game time
        return service


select_queue_premium = 35


def select_queue(node_id):
    if node_id == TICKET_QUEUE:
        selectStream(select_node_random)
        r = random()
        if r <= p_size:  # green pass
            node_list[node_id].priority_arrival = True  # green pass
            return
        else:
            node_list[node_id].priority_arrival = False  # test
            return
    else:
        selectStream(select_queue_premium)
        r = random()
        if r <= p_premium:
            node_list[node_id].priority_arrival = True  # premium ticket
            return
        else:
            node_list[node_id].priority_arrival = False  # standard ticket
            return


select_node_positive = 75


def is_positive():
    selectStream(select_node_positive)
    r = random()
    if r <= p_positive:
        return True
    else:
        return False


if __name__ == '__main__':
    # settings
    node_list = []
    for i in range(nodes + 1):
        if i == 0:
            node_list.append(SystemNode(i))
        else:
            node_list.append(StatusNode(i))
    plantSeeds(seed)

    time.current = START
    arrival = START  # global temp var for getArrival function     [minutes]

    # initialization of the first arrival event
    set_arrival_time(arrival_time_night)
    arrival += get_arrival(arrival_time)
    node = node_list[select_node(False)]  # node in cui schedulare l'arrivo
    select_queue(node.id)  # discriminazione della coda relativa all'arrivo
    node.arrival = arrival
    min_arrival = arrival
    count_global_number = 0.0

    while (min_arrival < STOP) or (node_list[0].number > 0):

        node_to_process = node_list[next_event()]  # node with minimum arrival or completion time
        time.next = minimum(node_to_process.arrival, node_to_process.completion)
        # Aggiornamento delle aree basate sul giro prima
        for i in range(0, len(node_list)):
            if i != 0:
                if node_list[i].priority_completion:
                    if node_list[i].more_p_stat.number > 0:
                        node_list[i].more_p_stat.node += (time.next - time.current) * node_list[i].more_p_stat.number
                        node_list[i].more_p_stat.queue += (time.next - time.current) * (
                                    node_list[i].more_p_stat.number - 1)
                        node_list[i].more_p_stat.service += (time.next - time.current)
                    if node_list[i].less_p_stat.number > 0:
                        count_global_number += 1
                        node_list[i].less_p_stat.node += (time.next - time.current) * (
                                    node_list[i].less_p_stat.number - 1)
                        node_list[i].less_p_stat.queue += (time.next - time.current) * (
                                    node_list[i].less_p_stat.number - 1)
                        node_list[i].less_p_stat.service += (time.next - time.current)
                else:
                    if node_list[i].more_p_stat.number > 0:
                        node_list[i].more_p_stat.node += (time.next - time.current) * (
                                    node_list[i].more_p_stat.number - 1)
                        node_list[i].more_p_stat.queue += (time.next - time.current) * (
                                    node_list[i].more_p_stat.number - 1)
                        node_list[i].more_p_stat.service += (time.next - time.current)
                    if node_list[i].less_p_stat.number > 0:
                        count_global_number += 1
                        node_list[i].less_p_stat.node += (time.next - time.current) * (node_list[i].less_p_stat.number)
                        node_list[i].less_p_stat.queue += (time.next - time.current) * (
                                    node_list[i].less_p_stat.number - 1)
                        node_list[i].less_p_stat.service += (time.next - time.current)
            else:
                if node_list[i].number > 0:
                    node_list[i].stat.node += (time.next - time.current) * node_list[i].number
                    node_list[i].stat.queue += (time.next - time.current) * (node_list[i].number - 1)
                    node_list[i].stat.service += (time.next - time.current)

        current_for_update = time.current
        time.current = time.next  # advance the clock

        if time.current == node_to_process.arrival:
            # Set arrival time
            day = (time.current / 1440.0) // 1
            current_lambda = time.current - day * 1440.0

            if 480.0 <= current_lambda < 720.0:  # 8-12
                set_arrival_time(arrival_time_morning)
            elif 720.0 <= current_lambda < 1020.0:  # 12-17
                set_arrival_time(arrival_time_afternoon)
            elif 1020.0 <= current_lambda < 1320.0:  # 17-22
                set_arrival_time(arrival_time_evening)
            else:  # 22-8
                set_arrival_time(arrival_time_night)

            if node_to_process.priority_arrival is True:  # vediamo su quale coda è stato schedulato l'arrivo che stiamo processando
                node_to_process.more_p_stat.number += 1
            else:
                node_to_process.less_p_stat.number += 1

            node_list[0].number += 1  # update system stat
            arrival += get_arrival(arrival_time)
            node_selected_pos = select_node(False)
            select_queue(node_selected_pos)  # discriminazione della coda relativa all'arrivo
            # Se il prossimo arrivo è su un altro centro, bisogna eliminare l'arrivo sul centro processato altrimenti
            # sarà sempre il minimo
            if node_selected_pos != node_to_process.id:
                node_to_process.arrival = INFINITY
            node = node_list[node_selected_pos]

            if node.arrival != INFINITY:
                if node.priority_arrival is True:
                    node.more_p_stat.last = node.arrival
                else:
                    node.less_p_stat.last = node.arrival

                max_last = maximum(node.more_p_stat.last, node.less_p_stat.last)
                if node.more_p_stat.last is not None and node_list[0].last is not None and node_list[0].last < max_last:
                    node_list[0].last = max_last

            # update node and system last arrival time

            # Controllo che l'arrivo sul nodo i-esimo sia valido. In caso negativo
            # imposto come ultimo arrivo del nodo i-esimo l'arrivo precedentemente
            # considerato
            if arrival > STOP:
                if node.arrival != INFINITY:
                    if node.priority_arrival is True:
                        node.more_p_stat.last = node.arrival
                    else:
                        node.less_p_stat.last = node.arrival
                # update node and system last arrival time
                max_last = maximum(node.more_p_stat.last, node.less_p_stat.last)
                if node_list[0].last < max_last:
                    node_list[0].last = max_last
                node.arrival = INFINITY
            else:
                node.arrival = arrival

            # caso sistema completamente vuoto prima dell'arrivo del job su node_to_process
            if node_to_process.more_p_stat.number == 1 and node_to_process.less_p_stat.number == 0:
                node_to_process.priority_completion = True
                node_to_process.completion = time.current + get_service(node_to_process.id)
            elif node_to_process.more_p_stat.number == 0 and node_to_process.less_p_stat.number == 1:
                node_to_process.priority_completion = False
                node_to_process.completion = time.current + get_service(node_to_process.id)
        else:
            if node_to_process.priority_completion is True:
                node_to_process.more_p_stat.index += 1  # node stats update
                node_to_process.more_p_stat.number -= 1
            else:
                node_to_process.less_p_stat.index += 1  # node stats update
                node_to_process.less_p_stat.number -= 1

            if node_to_process.id != TICKET_QUEUE:  # system stats update
                node_list[0].index += 1
                node_list[0].number -= 1

            if node_to_process.more_p_stat.number > 0:
                node_to_process.priority_completion = True
                node_to_process.completion = time.current + get_service(node_to_process.id)
            elif node_to_process.less_p_stat.number > 0:
                node_to_process.priority_completion = False
                node_to_process.completion = time.current + get_service(node_to_process.id)
            else:
                node_to_process.completion = INFINITY

            if node_to_process.id == TICKET_QUEUE:  # a completion on TICKET_QUEUE trigger an arrival on ARCADE_i
                if not is_positive():

                    arcade_node = node_list[select_node(True)]  # on first global stats
                    select_queue(arcade_node.id)

                    # Update partial stats for arcade nodes
                    if arcade_node.priority_arrival is True:
                        arcade_node.more_p_stat.number += 1  # system stats don't updated
                        arcade_node.more_p_stat.last = time.current
                    else:
                        arcade_node.less_p_stat.number += 1  # system stats don't updated
                        arcade_node.less_p_stat.last = time.current
                    if arcade_node.more_p_stat.number == 1 and arcade_node.less_p_stat.number == 0:
                        arcade_node.priority_completion = True
                        arcade_node.completion = time.current + get_service(arcade_node.id)
                    elif arcade_node.more_p_stat.number == 0 and arcade_node.less_p_stat.number == 1:
                        arcade_node.priority_completion = False
                        arcade_node.completion = time.current + get_service(arcade_node.id)
                else:
                    node_list[0].index += 1
                    node_list[0].number -= 1

        arrival_list = [node_list[n].arrival for n in range(1, len(node_list))]
        min_arrival = sorted(arrival_list, key=lambda x: (x is None, x))[0]

    print(count_global_number)
    for i in range(0, len(node_list)):
        if i == 0:

            print("\n\nNode " + str(i))
            print("\nfor {0} jobs".format(node_list[i].index))
            print("   average interarrival time = {0:6.6f}".format(node_list[i].last / node_list[i].index))
            print("   average wait ............ = {0:6.6f}".format(node_list[i].stat.node / node_list[i].index))
            print("   average delay ........... = {0:6.6f}".format(node_list[i].stat.queue / node_list[i].index))
            print("   average # in the node ... = {0:6.6f}".format(node_list[i].stat.node / time.current))
            print("   average # in the queue .. = {0:6.6f}".format(node_list[i].stat.queue / time.current))
            print("   utilization ............. = {0:6.6f}".format(node_list[i].stat.service / time.current))
            print(node_list[i].last)
        else:
            tot_index = node_list[i].more_p_stat.index + node_list[i].less_p_stat.index
            print("\n\nNode " + str(i) + " priority queue")
            print("\nfor {0} jobs".format(node_list[i].more_p_stat.index))
            if node_list[i].more_p_stat.index != 0:
                print("   average interarrival time = {0:6.6f}".format(
                    node_list[i].more_p_stat.last / node_list[i].more_p_stat.index))
                print("   average wait ............ = {0:6.6f}".format(
                    node_list[i].more_p_stat.node / node_list[i].more_p_stat.index))
                print("   average delay ........... = {0:6.6f}".format(
                    node_list[i].more_p_stat.queue / node_list[i].more_p_stat.index))
                print("   average # in the node ... = {0:6.6f}".format(node_list[i].more_p_stat.node / time.current))
                print("   average # in the queue .. = {0:6.6f}".format(node_list[i].more_p_stat.queue / time.current))
                print("   utilization ............. = {0:6.6f}".format(node_list[i].more_p_stat.service / time.current))
                print(node_list[i].more_p_stat.last, node_list[i].more_p_stat.index)

            print("\n\nNode " + str(i) + " NON priority queue")
            print("\nfor {0} jobs".format(node_list[i].less_p_stat.index))
            if node_list[i].less_p_stat.index != 0:
                print("   average interarrival time = {0:6.6f}".format(
                    node_list[i].less_p_stat.last / node_list[i].less_p_stat.index))
                print("   average wait ............ = {0:6.6f}".format(
                    node_list[i].less_p_stat.node / node_list[i].less_p_stat.index))
                print("   average delay ........... = {0:6.6f}".format(
                    node_list[i].less_p_stat.queue / node_list[i].less_p_stat.index))
                print("   average # in the node ... = {0:6.6f}".format(node_list[i].less_p_stat.node / time.current))
                print("   average # in the queue .. = {0:6.6f}".format(node_list[i].less_p_stat.queue / time.current))
                print("   utilization ............. = {0:6.6f}".format(node_list[i].less_p_stat.service / time.current))
                print(node_list[i].less_p_stat.last, node_list[i].less_p_stat.index)
