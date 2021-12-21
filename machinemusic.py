"""
tempo is given in microseconds per beat
see https://www.fourmilab.ch/webtools/midicsv/#Download for formats

output header should be:
0, 0, Header, 1, 2, 120

where 1 is the file format,
2 is the number of tracks (1 for tempo and stuff, 1 for notes),

"""

import os
import numpy as np
import numpy.random as r
np.seterr(all='raise')
r.seed(4685)

'''
a = np.array([[r.random() for j in range(5)] for k in range(1 + 1)])
b = np.array([[r.random() for j in range(5)] for k in range(5 + 1)])
c = np.array([[r.random() for j in range(5)] for k in range(5 + 1)])
input = np.array(0.4)
mid = sig(input.dot(a[0]) + a[1])
m2 = sig(mid.dot(b[:-1]) + b[-1])
out = sig(m2.dot(c[:-1]) + c[-1])
'''


def CalcHiddenLayer1(input_num, output_num):
    return int(np.round(np.sqrt((output_num + 2) * input_num) + 2 * np.sqrt(input_num / (output_num + 2))))


def CalcHiddenLayer2(input_num, output_num):
    return int(np.round(output_num * np.sqrt(input_num / (output_num + 2))))


def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except FloatingPointError:
        print('Very large numbers are being used; something is probably wrong.')
        return x*0.0


def sigmoid_dx(x):
    try:
        return np.exp(-x) / ((1 + np.exp(-x))*(1 + np.exp(-x)))
    except FloatingPointError:
        print('Very large numbers are being used; something is probably wrong.')
        return x*0.0


# Network Layer Calculations using matrix multiplication
def NLC(vector, matrix):
    return sigmoid(vector.dot(matrix[:-1]) + matrix[-1])


for midi_name in os.listdir('/home/calvin/Desktop/machine_music/midirags'):
    os.system('./midicsv-1.1/midicsv midirags/' + midi_name + " csvrags/" + midi_name[:len(midi_name) - 4] + '_csv.txt')

note_min = 60
note_max = 60
note_count_max = 0
time_min = 0  # will always be 0
time_max = 0
total_files = 0
max_time_between_note_starts = 0
max_total_time = 0

# loop through once to find min and max values and stuff
for file_name in os.listdir('/home/calvin/Desktop/machine_music/csvrags'):
    file = open('csvrags/' + file_name, 'r', errors='ignore')

    temp_len = 0
    # note that 60 is c4 for note and on=1 while off=0

    nl = file.readline()
    note_queue = []
    prev_note_start = 0
    first_check = True

    while nl != '':
        if "Note_" in nl:
            split_list = nl.split(", ")
            note = int(split_list[4])
            time = int(split_list[1])

            if split_list[5] == '0\n' or split_list[2] == 'Note_off_c':
                queue_check = 0
                while note_queue[queue_check][0] != note:
                    queue_check += 1
                time_check = time - note_queue[queue_check][1]
                if time_check > time_max:
                    time_max = time_check
                note_queue.remove((note, note_queue[queue_check][1]))
                temp_len += 1
            else:
                if note > note_max:
                    note_max = note
                if note < note_min:
                    note_min = note
                note_queue.append((note, time))

                if first_check:
                    first_check = False
                elif time - prev_note_start > max_time_between_note_starts:
                    max_time_between_note_starts = time - prev_note_start
                prev_note_start = time

                if time > max_total_time:
                    max_total_time = time

        nl = file.readline()

    if temp_len > note_count_max:
        note_count_max = temp_len
    file.close()
    total_files += 1

note_min -= 5
note_max += 6
note_range = note_max - note_min + 1  # extra +1 is if no note is being played used to signify end of song

print("Lowest note:", note_min)
print("Highest note:", note_max)
print("Note range:", note_range)
print("Shortest time:", time_min)
print("Longest time:", time_max)
print("Max total note count:", note_count_max)
print("Number of files:", total_files)
print("Max time between note starts:", max_time_between_note_starts)
print("Max total time:", max_total_time)
print()
num_lines = 3

all_data = []
# now loop through to get data into an np array
for file_name in os.listdir('/home/calvin/Desktop/machine_music/csvrags'):
    print(file_name)
    file = open('csvrags/' + file_name, 'r', errors='ignore')
    temp_data = np.zeros((num_lines, note_count_max))

    nl = file.readline()

    # data stored as [time0, time1, ...], [note0, note1, ...], [length0, length1, ...]
    j = 0
    note_queue = []
    while nl != '':
        if "Note_" in nl:
            split_list = nl.split(", ")
            time_stamp = int(split_list[1])
            note = int(split_list[4]) - note_min + 1
            if split_list[5] == '0\n' or split_list[2] == 'Note_off_c':
                queue_check = 0
                while note_queue[queue_check][0] != note:
                    queue_check += 1
                note_len = time_stamp - note_queue[queue_check][1]
                temp_data[2][note_queue[queue_check][2]] = note_len
                note_queue.remove((note, note_queue[queue_check][1], note_queue[queue_check][2]))
            else:
                temp_data[0][j] = time_stamp
                temp_data[1][j] = note
                note_queue.append((note, time_stamp, j))
                j += 1

        nl = file.readline()

    file.close()

    all_data.append(temp_data)

    # below creates copies for more training data, transposed
    for k in range(5):
        temp_transpose_data = np.zeros((num_lines, note_count_max))
        entry = 0
        while entry != note_count_max and temp_data[1][entry] != 0:
            temp_transpose_data[0][entry] = temp_data[0][entry] - 1 - k
            temp_transpose_data[1][entry] = temp_data[1][entry]
            temp_transpose_data[2][entry] = temp_data[2][entry]
            entry += 1
        all_data.append(temp_transpose_data)

    for k in range(6):
        temp_transpose_data = np.zeros((num_lines, note_count_max))
        entry = 0
        while entry != note_count_max and temp_data[1][entry] != 0:
            temp_transpose_data[0][entry] = temp_data[0][entry] + 1 + k
            temp_transpose_data[1][entry] = temp_data[1][entry]
            temp_transpose_data[2][entry] = temp_data[2][entry]
            entry += 1
        all_data.append(temp_transpose_data)


# I believe it works properly! Tempo may need to be adjusted though
def OutputNPArrayToFile(note_arr, name):
    out = open('outputcsv/' + name + '_csv.txt', 'w')
    out.write("""0, 0, Header, 1, 2, 200
1, 0, Start_track
1, 0, Title_t, "GO!"
1, 0, Time_signature, 4, 2, 24, 8
1, 0, Key_signature, 0, "major"
1, 0, Tempo, 350000
1, 0, End_track
2, 0, Start_track\n""")

    end_time = 0  # assigned here to be used later
    i = 0
    end_note_queue = []  # entries are of format (end_note_timestamp, note)
    while i != note_count_max and note_arr[1][i] != 0:
        t = int(note_arr[0][i])             # timestamp
        n = int(note_arr[1][i] + note_min - 1)  # note
        l = int(note_arr[2][i])             # note length
        while len(end_note_queue) > 0 and end_note_queue[0][0] < t:
            end_time, end_note = end_note_queue.pop(0)
            out.write('2, ' + str(end_time) + ', Note_on_c, 0, ' + str(end_note) + ', 0\n')

        out.write('2, ' + str(t) + ', Note_on_c, 0, ' + str(n) + ', 80\n')
        end_note_queue.append((t + l, n))
        end_note_queue.sort()
        i += 1
    while len(end_note_queue) > 0:
        end_time, end_note = end_note_queue.pop(0)
        out.write('2, ' + str(end_time) + ', Note_on_c, 0, ' + str(end_note) + ', 0\n')

    out.write("""2, """ + str(end_time + 100) + """, End_track
0, 0, End_of_file""")
    out.close()

    os.system('./midicsv-1.1/csvmidi outputcsv/' + name + '_csv.txt newmidis/' + name + ".mid")


total_songs = len(all_data)
print("\nData Files:", total_songs)

# for i in range(total_songs):
#    OutputNPArrayToFile(all_data[i], str(i//12) + '_' + str(i % 12))

# Create a generator and a discriminator neural network
# note the + 1 comes from the bias weight at the end

print('\nCreating neural networks...')

# number of outputs of neural network for generator, inputs for discriminator
rag_data_len = num_lines * note_count_max

# should both be 123
gen_hid_nod0 = CalcHiddenLayer1(1, rag_data_len)
gen_hid_nod1 = CalcHiddenLayer2(1, rag_data_len)
gen_net = {'h0': np.array([[2*(r.random() - 0.5) for j in range(gen_hid_nod0)] for k in range(1 + 1)],
                          dtype=np.float128),
           'h1': np.array([[2*(r.random() - 0.5) for j in range(gen_hid_nod1)] for k in range(gen_hid_nod0 + 1)],
                          dtype=np.float128),
           'o':  np.array([[2*(r.random() - 0.5) for j in range(rag_data_len)] for k in range(gen_hid_nod1 + 1)],
                          dtype=np.float128)}

# should be 355 and 71 respectively
dis_hid_nod0 = CalcHiddenLayer1(rag_data_len, 1)
dis_hid_nod1 = CalcHiddenLayer2(rag_data_len, 1)
dis_net = {'h0': np.array([[2*(r.random() - 0.5) for j in range(dis_hid_nod0)] for k in range(rag_data_len + 1)],
                          dtype=np.float128),
           'h1': np.array([[2*(r.random() - 0.5) for j in range(dis_hid_nod1)] for k in range(dis_hid_nod0 + 1)],
                          dtype=np.float128),
           'o':  np.array([[2*(r.random() - 0.5) for j in range(1)]            for k in range(dis_hid_nod1 + 1)],
                          dtype=np.float128)}
print('done!\n')


# Generate music:
print('Generating music...')
gen_music = []
for k in range(total_songs):
    song_seed = np.array(2000*(r.random() - 0.5))
    song_vector = NLC(NLC(NLC(song_seed, gen_net['h0']), gen_net['h1']), gen_net['o'])[0]

    time_stamp_vector = song_vector[:note_count_max]
    # starts 200 'ticks' in (doesn't matter)
    time_stamp_vector[0] = np.round(time_stamp_vector[0]*max_time_between_note_starts + 200)
    for j in range(len(time_stamp_vector) - 1):
        time_stamp_vector[j+1] = np.round(time_stamp_vector[j+1]*max_time_between_note_starts + time_stamp_vector[j])

    output_song = np.array([time_stamp_vector,
                            np.round(note_range * song_vector[note_count_max:2*note_count_max]),
                            np.round(time_max * song_vector[-note_count_max:])], dtype=np.float128)

    OutputNPArrayToFile(output_song, str(k))
    gen_music.append(output_song)


# Discriminate music:
def flattenArray(np_array):
    return np.array([np.append(np.append(np_array[0]/(max_total_time*30), np_array[1]/note_range),
                               np_array[2]/time_max)])


learning_rate = 0.5
all_data.extend(gen_music)
gen_total_songs = 2*total_songs

while(True):
    error = gen_total_songs + 1  # error can only go up to 1
    print('\nstarting discriminator training...')
    while error / gen_total_songs > 0.4:  # FIXME change to 0.25 eventually (probably, should research)
        count = 0
        error = 0
        o_weight_gradient = np.zeros((dis_hid_nod1, 1), dtype=np.float128)
        h1_weight_gradient = np.zeros((dis_hid_nod0, dis_hid_nod1), dtype=np.float128)
        h0_weight_gradient = np.zeros((rag_data_len, dis_hid_nod0), dtype=np.float128)
        o_bias_gradient = np.zeros(1, dtype=np.float128)
        h1_bias_gradient = np.zeros(dis_hid_nod1, dtype=np.float128)
        h0_bias_gradient = np.zeros(dis_hid_nod0, dtype=np.float128)
        target_output = 1  # when song is real
        for song in all_data:
            flat_song = flattenArray(song)
            if count == total_songs:
                target_output = 0  # when song is generated
            node_layer0 = NLC(flat_song, dis_net['h0'])
            node_layer1 = NLC(node_layer0, dis_net['h1'])
            actual_output = NLC(node_layer1, dis_net['o'])[0][0]
            error += (actual_output - target_output) * (actual_output - target_output)

            node_layer1t = node_layer1.transpose()
            node_layer0t = node_layer0.transpose()
            song_t = flat_song.transpose()

            sig_der_o = sigmoid_dx(np.sum(dis_net['o'][:-1] * node_layer1t) + dis_net['o'][-1])
            err_der = (actual_output - target_output) * learning_rate * sig_der_o  # error derivative and other terms

            # Below is a 71 length vector
            o_weight_gradient += err_der * node_layer1t
            o_bias_gradient += err_der

            sig_der_h1 = sigmoid_dx(sum(dis_net['h1'][:-1] * node_layer0t) + dis_net['h1'][-1])

            temp = sig_der_h1 * err_der * dis_net['o'][:-1].transpose()
            # below is a 355 by 71 matrix
            h1_weight_gradient += temp * node_layer0t
            h1_bias_gradient += temp[0]

            sig_der_h0 = sigmoid_dx(sum(dis_net['h0'][:-1] * song_t) + dis_net['h0'][-1])
            temp2 = sig_der_h0 * err_der * np.dot(dis_net['h1'][:-1] * sig_der_h1, dis_net['o'][:-1]).transpose()
            h0_weight_gradient += temp2 * song_t
            h0_bias_gradient += temp2[0]

            count += 1

        print("Average Error:", error / gen_total_songs)
        dis_net['o'][-1] -= o_bias_gradient / count
        dis_net['h1'][-1] -= h1_bias_gradient / count
        dis_net['h0'][-1] -= h0_bias_gradient / count
        dis_net['o'][:-1] -= o_weight_gradient / count
        dis_net['h1'][:-1] -= h1_weight_gradient / count
        dis_net['h0'][:-1] -= h0_weight_gradient / count


    print()
    print("starting generator training...")
    target_output = 1  # when the Discriminator thinks the song is real (aka not generated)
    output_songs = [None for k in range(total_songs)]
    error = total_songs
    while error / total_songs > 0.25:
        output_songs_count = 0
        gen_music = []
        song_seeds = [np.array(2000*(r.random() - 0.5)) for k in range(total_songs)]
        error = 0

        o_weight_gradient = np.zeros((gen_hid_nod1, rag_data_len), dtype=np.float128)
        h1_weight_gradient = np.zeros((gen_hid_nod0, gen_hid_nod1), dtype=np.float128)
        h0_weight_gradient = np.zeros((1, gen_hid_nod0), dtype=np.float128)
        o_bias_gradient = np.zeros(rag_data_len, dtype=np.float128)
        h1_bias_gradient = np.zeros(gen_hid_nod1, dtype=np.float128)
        h0_bias_gradient = np.zeros(gen_hid_nod0, dtype=np.float128)

        for some_song in song_seeds:
            # Generator vector nodes
            song_vec_h0 = NLC(some_song, gen_net['h0'])
            song_vec_h1 = NLC(song_vec_h0, gen_net['h1'])
            song_vec_o  = NLC(song_vec_h1, gen_net['o'])
            song_vector = song_vec_o[0]

            # Processing output from Gen to input to Dis
            time_stamp_vector = song_vector[:note_count_max]
            # starts 200 'ticks' in (doesn't matter)
            time_stamp_vector[0] = np.round(time_stamp_vector[0]*max_time_between_note_starts + 200)
            for j in range(len(time_stamp_vector) - 1):
                time_stamp_vector[j+1] = np.round(time_stamp_vector[j+1]*max_time_between_note_starts + time_stamp_vector[j])
            output_song = np.array([time_stamp_vector,
                                    np.round(note_range * song_vector[note_count_max:2*note_count_max]),
                                    np.round(time_max * song_vector[-note_count_max:])], dtype=np.float128)

            output_songs[output_songs_count] = output_song
            output_songs_count += 1
            flat_song = flattenArray(output_song)  # input to Dis

            # Discriminator vector nodes
            node_layer0 = NLC(flat_song, dis_net['h0'])
            node_layer1 = NLC(node_layer0, dis_net['h1'])
            actual_output = NLC(node_layer1, dis_net['o'])[0][0]
            error += (actual_output - target_output) * (actual_output - target_output)

            node_layer0t = node_layer0.transpose()
            node_layer1t = node_layer1.transpose()
            song_t = flat_song.transpose()

            sig_der_o = sigmoid_dx(np.sum(dis_net['o'][:-1] * node_layer1t) + dis_net['o'][-1])

            err_der = (actual_output - target_output) * learning_rate * sig_der_o  # error derivative and other terms
            sig_der_h1 = sigmoid_dx(sum(dis_net['h1'][:-1] * node_layer0t) + dis_net['h1'][-1])
            sig_der_h0 = sigmoid_dx(sum(dis_net['h0'][:-1] * song_t) + dis_net['h0'][-1])

            song_h0t = song_vec_h0.transpose()
            song_h1t = song_vec_h1.transpose()

            gen_sig_der_o = sigmoid_dx(sum(gen_net['o'][:-1] * song_h1t) + gen_net['o'][-1])  # 7728 elements (1D)
            temp_o = gen_sig_der_o * np.dot(sig_der_h0 * dis_net['h0'][:-1], np.dot(sig_der_h1 * dis_net['h1'][:-1], err_der * dis_net['o'][:-1])).transpose()
            o_weight_gradient += song_h1t * temp_o
            o_bias_gradient += temp_o[0]

            gen_sig_der_h1 = sigmoid_dx(sum(gen_net['h1'][:-1] * song_h0t) + gen_net['h1'][-1])
            temp_h1 = gen_sig_der_h1 * np.dot(gen_sig_der_o * gen_net['o'][:-1], np.dot(sig_der_h0 * dis_net['h0'][:-1], np.dot(sig_der_h1 * dis_net['h1'][:-1], err_der * dis_net['o'][:-1]))).transpose()
            h1_weight_gradient += song_h0t * temp_h1
            h1_bias_gradient += temp_h1[0]

            gen_sig_der_h0 = sigmoid_dx(sum(gen_net['h0'][:-1] * some_song) + gen_net['h0'][-1])
            temp_h0 = gen_sig_der_h0 * np.dot(gen_sig_der_h1 * gen_net['h1'][:-1], np.dot(gen_sig_der_o * gen_net['o'][:-1], np.dot(sig_der_h0 * dis_net['h0'][:-1], np.dot(sig_der_h1 * dis_net['h1'][:-1], err_der * dis_net['o'][:-1])))).transpose()
            h0_weight_gradient += some_song * temp_h0
            h0_bias_gradient += temp_h0[0]

        print("Average Gen Error:", error / total_songs)
        gen_net['o'][-1] -= o_bias_gradient / total_songs
        gen_net['h1'][-1] -= h1_bias_gradient / total_songs
        gen_net['h0'][-1] -= h0_bias_gradient / total_songs
        gen_net['o'][:-1] -= o_weight_gradient / total_songs
        gen_net['h1'][:-1] -= h1_weight_gradient / total_songs
        gen_net['h0'][:-1] -= h0_weight_gradient / total_songs

    for k in range(total_songs):
        OutputNPArrayToFile(output_songs[k], str(k))
    all_data[total_songs:] = output_songs
