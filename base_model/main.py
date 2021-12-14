from utils.rngs import random, selectStream, plantSeeds
from utils.rvgs import Exponential, TruncatedNormal

nodes = 4  # n nodi
arrival_time = 0.0
arrival_time_morning = 15.0
arrival_time_afternoon = 5.0
arrival_time_evening = 15.0
arrival_time_night = 25.0

seed = 123456789  # TODO: Controlla il seed migliore o forse era il multiplier?
START = 0.0
STOP = 1 * 1 * 28 * 1440.0  # Minutes
INFINITY = STOP * 100.0
p_ticket_queue = 0.8
TICKET_QUEUE = 1
# ARCADE1 = 2
# ARCADE2 = 3
# ARCADE3 = 4
p_size = 0.6


class Track:
    node = 0.0  # time integrated number in the node
    queue = 0.0  # time integrated number in the queue
    service = 0.0  # time integrated number in service

    def __init__(self):
        self.node = 0.0
        self.queue = 0.0
        self.service = 0.0


class Time:
    current = None  # current time
    next = None  # next (most imminent) event time


time = Time()


class StatusNode:
    id = None
    arrival = None  # next arrival time
    completion = None  # next completion time
    last = 0.0  # last arrival time
    index = 0.0  # jobs departed
    number = 0.0  # jobs in node
    stat = None

    def __init__(self, id_node):
        self.id = id_node
        self.stat = Track()  # Track stats


def set_arrival_time(x):
    global arrival_time
    arrival_time = x


def get_arrival_time():
    return arrival_time


arr_est = 0


def select_node(from_tkt_queue):
    selectStream(5)  # TODO: scegliere gli streams sequenziali o no?
    if from_tkt_queue:
        r = random()
        for i in range(1, nodes):
            if r <= i / (nodes - 1):
                return i + 1
    # if r <= 1 / (nodes - 1):
    #     return ARCADE1
    # elif r <= 2 / (nodes - 1):
    #     return ARCADE2
    # else:
    #     return ARCADE3

    # Caso arrivo dall'esterno

    r = random()
    if r <= p_ticket_queue:
        global arr_est
        arr_est += 1
        # print(arr_est)
        return TICKET_QUEUE
    else:
        r = random()
        for i in range(1, nodes):
            if r <= i / (nodes - 1):
                return i + 1
    #  if r <= 1.0 / float(nodes - 1):
    #      # print(1.0 / float(nodes - 1))
    #      return ARCADE1
    #  elif r <= 2.0 / float(nodes - 1):
    #      # print(2.0 / float(nodes - 1))
    #      return ARCADE2
    #  else:
    #      return ARCADE3


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
    time_event = []
    for i in range(1, len(node_list)):
        time_event.append(node_list[i].arrival)
        time_event.append(node_list[i].completion)

    time_event = sorted(time_event, key=lambda x: (x is None, x))

    for i in range(1, len(time_event)):
        if time_event[0] == node_list[i].arrival or time_event[0] == node_list[i].completion:
            return i


# TODO: Provare l'esponenziale troncata e verificare i tempi di interarrivo
def get_arrival(y):
    # ---------------------------------------------
    # * generate the next arrival time from an Exponential distribution.
    # * --------------------------------------------

    selectStream(0)
    return Exponential(y)


def get_service(id_node):
    # --------------------------------------------
    # * generate the next service time
    # * --------------------------------------------
    # */
    if id_node == TICKET_QUEUE:
        selectStream(6)
        r = random()
        if r <= p_size:  # green pass
            selectStream(id_node + 10)
            service = TruncatedNormal(2, 1.5, 1, 3)  # green pass
            # print("Green pass: ", service)
            return service
        else:
            selectStream(id_node + 15)
            service = TruncatedNormal(10, 1.5, 8, 12)  # covid test
            # print("covid test: ", service)
            return service
    else:
        selectStream(id_node + 10)
        service = TruncatedNormal(15, 3, 3, 25)  # arcade game time
        # print("arcade game time: ", service)
        return service


if __name__ == '__main__':
    # settings
    times = [0, 0, 0, 0]
    node_list = [StatusNode(i) for i in range(nodes + 1)]  # in 0 global stats
    plantSeeds(seed)

    time.current = START
    arrival = START  # global temp var for getArrival function     [minutes]

    # initialization of the first arrival event
    set_arrival_time(arrival_time_night)
    times[3] += 1
    arrival += get_arrival(arrival_time)
    node = node_list[select_node(False)]
    node.arrival = arrival
    min_arrival = arrival

    while (min_arrival < STOP) or (node_list[0].number > 0):

        node_to_process = node_list[next_event()]  # node with minimum arrival or completion time
        time.next = minimum(node_to_process.arrival, node_to_process.completion)
        # Aggiornamento delle aree basate sul giro prima
        for i in range(0, len(node_list)):
            if node_list[i].number > 0:
                # if i == 0 or i == node_to_process.id:
                node_list[i].stat.node += (time.next - time.current) * node_list[i].number
                node_list[i].stat.queue += (time.next - time.current) * (node_list[i].number - 1)
                node_list[i].stat.service += (time.next - time.current)

        current_for_update = time.current
        time.current = time.next  # advance the clock

        if time.current == node_to_process.arrival:
            # TODO: Se sto finto switch lo metto fuori le stastiche cambiano..bah
            # Set arrival time
            day = (time.current / 1440.0) // 1
            current_lambda = time.current - day * 1440.0

            if 480.0 <= current_lambda < 720.0:  # 8-12
                set_arrival_time(arrival_time_morning)
                times[0] += 1
            elif 720.0 <= current_lambda < 1020.0:  # 12-17
                set_arrival_time(arrival_time_afternoon)
                times[1] += 1
            elif 1020.0 <= current_lambda < 1320.0:  # 17-22
                set_arrival_time(arrival_time_evening)
                times[2] += 1
            else:  # 22-8
                set_arrival_time(arrival_time_night)
                times[3] += 1
            # print("day: ", day, ", current_lambda: ", current_lambda, ", t_current: ", time.current,
            # ", lambda: ", get_arrival_time())
            # print(day, time.current)
            node_to_process.number += 1
            node_list[0].number += 1  # update system stat
            arrival += get_arrival(arrival_time)
            node_selected_pos = select_node(False)

            # Se il prossimo arrivo è su un altro centro, bisogna eliminare l'arrivo sul centro processato altrimenti
            # sarà sempre il minimo
            if node_selected_pos != node_to_process.id:
                node_to_process.arrival = INFINITY
            node = node_list[node_selected_pos]

            if node.arrival != INFINITY:
                node.last = node.arrival
                if node.last is not None and node_list[0].last is not None and node_list[0].last < node.last:
                    node_list[0].last = node.last
            # update node and system last arrival time

            # Controllo che l'arrivo sul nodo i-esimo sia valido. In caso negativo
            # imposto come ultimo arrivo del nodo i-esimo l'arrivo precedentemente
            # considerato
            if arrival > STOP:
                if node.arrival != INFINITY:
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
                arcade_node = node_list[select_node(True)]  # on first global stats

                # Update partial stats for arcade nodes
                # if arcade_node.number > 0:
                #     arcade_node.stat.node += (time.next - current_for_update) * arcade_node.number
                #     arcade_node.stat.queue += (time.next - current_for_update) * (arcade_node.number - 1)
                #     arcade_node.stat.service += (time.next - current_for_update)

                arcade_node.number += 1  # system stats don't updated
                arcade_node.last = time.current

                if arcade_node.number == 1:
                    arcade_node.completion = time.current + get_service(arcade_node.id)

        arrival_list = [node_list[n].arrival for n in range(1, len(node_list))]
        min_arrival = sorted(arrival_list, key=lambda x: (x is None, x))[0]

    print(times)
    for i in range(0, len(node_list)):
        print(node_list[i].last)
        print("\n\nNode " + str(i))
        print("\nfor {0} jobs".format(node_list[i].index))
        print("   average interarrival time = {0:6.6f}".format(node_list[i].last / node_list[i].index))
        print("   average wait ............ = {0:6.6f}".format(node_list[i].stat.node / node_list[i].index))
        print("   average delay ........... = {0:6.6f}".format(node_list[i].stat.queue / node_list[i].index))
        print("   average # in the node ... = {0:6.6f}".format(node_list[i].stat.node / time.current))
        print("   average # in the queue .. = {0:6.6f}".format(node_list[i].stat.queue / time.current))
        print("   utilization ............. = {0:6.6f}".format(node_list[i].stat.service / time.current))
