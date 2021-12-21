import numpy as np
import numpy.random as r
import copy
import time

print("Starting...")
rag_data_len = 5035 * 3
dis_hid_nod0 = 355
dis_hid_nod1 = 71

r.seed(32234)


def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except FloatingPointError:
        print('hmmmm...')
        return x * 0.0


def sigmoid_dx(x):
    try:
        return np.exp(-x) / ((1 + np.exp(-x)) * (1 + np.exp(-x)))
    except FloatingPointError:
        print('hmmmm...')
        return x * 0.0


def NLC(vector, matrix):
    return sigmoid(vector.dot(matrix[:-1]) + matrix[-1])


dis_net = {'h0': np.array([[2 * (r.random() - 0.5) for j in range(dis_hid_nod0)] for k in range(rag_data_len + 1)],
                          dtype=np.float128),
           'h1': np.array([[2 * (r.random() - 0.5) for j in range(dis_hid_nod1)] for k in range(dis_hid_nod0 + 1)],
                          dtype=np.float128),
           'o': np.array([[2 * (r.random() - 0.5) for j in range(1)] for k in range(dis_hid_nod1 + 1)],
                         dtype=np.float128)}

dis_net2 = copy.deepcopy(dis_net)

runs = 101999999
learning_rate = 10
target_output = 0.5

songs = np.array([np.array([[r.random() for i in range(rag_data_len)]]) for j in range(3)])


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
start = time.time()
for sdfasf  in range(runs):
    error = 0
    count = 0
    o_weight_gradient = np.zeros((dis_hid_nod1, 1))
    h1_weight_gradient = np.zeros((dis_hid_nod0, dis_hid_nod1))
    h0_weight_gradient = np.zeros((rag_data_len, dis_hid_nod0))
    for song in songs:
        node_layer0 = NLC(song, dis_net2['h0'])
        node_layer1 = NLC(node_layer0, dis_net2['h1'])
        actual_output = NLC(node_layer1, dis_net2['o'])[0][0]
        error += (actual_output - 0.5) * (actual_output - 0.5)

        node_layer1t = node_layer1.transpose()
        node_layer0t = node_layer0.transpose()
        song_t = song.transpose()
        err_der = actual_output - target_output  # error derivative

        sig_der_o = sigmoid_dx(np.sum(dis_net2['o'][:-1] * node_layer1t) + dis_net2['o'][-1])
        # Below is a 71 length vector
        o_weight_gradient += learning_rate * err_der * sig_der_o * node_layer1t

        sig_der_h1 = sigmoid_dx(sum(dis_net2['h1'][:-1] * node_layer0t) + dis_net2['h1'][-1])

        # below is a 355 by 71 matrix
        h1_weight_gradient += learning_rate * sig_der_o * sig_der_h1 * node_layer0t * err_der * dis_net2['o'][:-1].transpose()

        sig_der_h0 = sigmoid_dx(sum(dis_net2['h0'][:-1] * song_t) + dis_net2['h0'][-1])
        h0_weight_gradient += learning_rate * sig_der_h0 * sig_der_o * song_t * err_der * np.dot(dis_net2['h1'][:-1] * sig_der_h1, dis_net2['o'][:-1]).transpose()


        count += 1

    print()
    print("Average Error:", error / len(songs))
    print(np.dot(dis_net2['h1'][:-1] * sig_der_h1, dis_net2['o'][:-1]).transpose().shape)
    #print(o_weight_gradient/count)
    #print(h1_weight_gradient/count)
    #print(h0_weight_gradient/count)
    #dis_net2['o'][:-1] -= o_weight_gradient/count
    #dis_net2['h1'][:-1] -= h1_weight_gradient/count
    dis_net2['h0'][:-1] -= h0_weight_gradient/count

