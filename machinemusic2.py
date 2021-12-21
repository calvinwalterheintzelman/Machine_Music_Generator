import os
import numpy as np
import numpy.random as r
np.seterr(all='raise')
r.seed(234221)


# Used to determine size of neural network
def CalcHiddenLayer1(input_num, output_num):
    return int(np.round(np.sqrt((output_num + 2) * input_num) + 2 * np.sqrt(input_num / (output_num + 2))))


# Used to determine size of neural network
def CalcHiddenLayer2(input_num, output_num):
    return int(np.round(output_num * np.sqrt(input_num / (output_num + 2))))


def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except FloatingPointError:
        return x*0.0


def sigmoid_dx(x):
    try:
        return np.exp(-x) / ((1 + np.exp(-x))*(1 + np.exp(-x)))
    except FloatingPointError:
        return x*0.0


# Network Layer Calculations using matrix multiplication
def NLC(vector, matrix):
    return sigmoid(vector.dot(matrix[:-1]) + matrix[-1])


for midi_name in os.listdir('/home/calvin/Desktop/machine_music/midirags'):
    os.system('./midicsv-1.1/midicsv midirags/' + midi_name + " csvrags/" + midi_name[:len(midi_name) - 4] + '_csv.txt')

note_min = 60  # lowest note
note_max = 60  # highest note
note_count_max = 0  # max total number of potential notes
total_files = 0  # number of files (duh)
time_min = 0  # shortest time between notes (will always be 0)
time_max = 0  # longest time between notes


# loop through once to find min and max values and stuff
for file_name in os.listdir('/home/calvin/Desktop/machine_music/csvrags'):
    file = open('csvrags/' + file_name, 'r', errors='ignore')

    temp_len = 0  # used to check total notes
    nl = file.readline()
    note_queue = []
    prev_note_start = 0
    while nl != '':
        if "Note_" in nl:
            split_list = nl.split(", ")
            note = int(split_list[4])
            time = int(split_list[1])

            if not(split_list[5] == '0\n' or split_list[2] == 'Note_off_c'):
                temp_len += 1
                if note > note_max:
                    note_max = note
                if note < note_min:
                    note_min = note
                time_between_notes_check = time - prev_note_start
                if time_between_notes_check > time_max:
                    time_max = time_between_notes_check
                prev_note_start = time

        nl = file.readline()

    if temp_len > note_count_max:
        note_count_max = temp_len
    file.close()
    total_files += 1

note_min -= 5
note_max += 6
note_range = note_max - note_min + 1  # extra +1 is if no note is being played used to signify end of song

print('Lowest note:', note_min)
print('Highest note:', note_max)
print('Note range:', note_range)
print('Shortest time between notes:', time_min)
print('Longest time between notes:', time_max)
print('Total notes:', note_count_max)
print('Total files:', total_files)
print()

num_lines = 2  # one for time between note, one for note
all_data = []
for file_name in os.listdir('/home/calvin/Desktop/machine_music/csvrags'):
    print(file_name)
    file = open('csvrags/' + file_name, 'r', errors='ignore')
    temp_data = np.zeros((num_lines, note_count_max))

    time = None
    nl = file.readline()
    j = 0
    prev_note_start = 0
    while nl != '':
        if "Note_" in nl:
            split_list = nl.split(", ")
            time = int(split_list[1])
            note = int(split_list[4]) - note_min + 1
            if not(split_list[5] == '0\n' or split_list[2] == 'Note_off_c'):
                temp_data[0][j] = note
                temp_data[1][j] = time - prev_note_start
                prev_note_start = time
                j += 1

        nl = file.readline()

    while j != note_count_max:
        temp_data[1][j] = time + 1
        j += 1

    file.close()
    all_data.append(temp_data)

    # Add transpoitions of each music piece to data
    for k in range(5):
        all_data.append(np.array([temp_data[0] - k - 1, temp_data[1]]))
    for k in range(6):
        all_data.append(np.array([temp_data[0] + k + 1, temp_data[1]]))


# Note tha tempo may need to be adjusted
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
    note_length = 80  # can change to be anything, really. The length of each note played
    i = 0
    time_stamp = 0
    note_end_queue = []
    while i < note_count_max and note_arr[0][i] != 0:
        n = note_arr[0][i] + note_min - 1
        t = note_arr[1][i] + time_stamp
        while len(note_end_queue) > 0 and note_end_queue[0][1] <= t:
            end_note, end_time = note_end_queue.pop(0)
            out.write('2, ' + str(end_time) + ', Note_on_c, 0, ' + str(end_note) + ', 0\n')

        time_stamp = t
        out.write('2, ' + str(t) + ', Note_on_c, 0, ' + str(n) + ', 80\n')
        note_end_queue.append((n, t + note_length))
        i += 1
    while len(note_end_queue) > 0:
        end_note, end_time = note_end_queue.pop(0)
        out.write('2, ' + str(end_time) + ', Note_on_c, 0, ' + str(end_note) + ', 0\n')

    out.write("""2, """ + str(end_time + 100) + """, End_track
0, 0, End_of_file""")
    out.close()

    os.system('./midicsv-1.1/csvmidi outputcsv/' + name + '_csv.txt newmidis/' + name + ".mid")


total_songs = len(all_data)
print("\nData Files:", total_songs)
print('\nCreating neural networks...')

# number of outputs of neural network for generator, inputs for discriminator
rag_data_len = num_lines * note_count_max  # should be 10070

# should both be 100
gen_hid_nod0 = CalcHiddenLayer1(1, rag_data_len)
gen_hid_nod1 = CalcHiddenLayer2(1, rag_data_len)
gen_net = {'h0': np.array([[2*(r.random() - 0.5) for j in range(gen_hid_nod0)] for k in range(1 + 1)],
                          dtype=np.float128),
           'h1': np.array([[2*(r.random() - 0.5) for j in range(gen_hid_nod1)] for k in range(gen_hid_nod0 + 1)],
                          dtype=np.float128),
           'o':  np.array([[2*(r.random() - 0.5) for j in range(rag_data_len)] for k in range(gen_hid_nod1 + 1)],
                          dtype=np.float128)}

# should be 290 and 58 respectively
dis_hid_nod0 = CalcHiddenLayer1(rag_data_len, 1)
dis_hid_nod1 = CalcHiddenLayer2(rag_data_len, 1)
dis_net = {'h0': np.array([[2*(r.random() - 0.5) for j in range(dis_hid_nod0)] for k in range(rag_data_len + 1)],
                          dtype=np.float128),
           'h1': np.array([[2*(r.random() - 0.5) for j in range(dis_hid_nod1)] for k in range(dis_hid_nod0 + 1)],
                          dtype=np.float128),
           'o':  np.array([[2*(r.random() - 0.5) for j in range(1)]            for k in range(dis_hid_nod1 + 1)],
                          dtype=np.float128)}
print('done!\n')

# Flatten all_data entries
for i in range(len(all_data)):
    all_data[i][0] = all_data[i][0] / note_range
    all_data[i][1] = all_data[i][1] / time_max
    all_data[i] = np.array([all_data[i].flatten()])

# Initially generate some junk music
for k in range(total_songs):
    song_seed = np.array(2 * (r.random() - 0.5))
    song_vector = NLC(NLC(NLC(song_seed, gen_net['h0']), gen_net['h1']), gen_net['o'])[0]
    output_song = np.array([np.round(note_range * song_vector[:note_count_max]),
                            np.round(time_max * song_vector[note_count_max:])])
    OutputNPArrayToFile(output_song, str(k))
    all_data.append(np.array([song_vector]))

gen_learning_rate = 0.5
dis_learning_rate = 0.01
gen_total_songs = 2*total_songs
output_songs = [None for k in range(total_songs)]
while True:
    print('\nStarting discriminator training...')
    rounds = 0
    error = gen_total_songs + 1  # error can only go up to 1
    while error / gen_total_songs > 0.25 and rounds < 24:  # FIXME change to 0.25 eventually (probably, should research)
        o_weight_gradient = np.zeros((dis_hid_nod1, 1), dtype=np.float128)
        h1_weight_gradient = np.zeros((dis_hid_nod0, dis_hid_nod1), dtype=np.float128)
        h0_weight_gradient = np.zeros((rag_data_len, dis_hid_nod0), dtype=np.float128)
        o_bias_gradient = np.zeros(1, dtype=np.float128)
        h1_bias_gradient = np.zeros(dis_hid_nod1, dtype=np.float128)
        h0_bias_gradient = np.zeros(dis_hid_nod0, dtype=np.float128)

        target_output = 1
        count = 0
        error = 0
        for song in all_data:
            if count >= total_songs:
                target_output = 0  # when song is generated
            node_layer0 = NLC(song, dis_net['h0'])
            node_layer1 = NLC(node_layer0, dis_net['h1'])
            actual_output = NLC(node_layer1, dis_net['o'])[0][0]
            error += (actual_output - target_output) * (actual_output - target_output)

            node_layer1t = node_layer1.transpose()
            node_layer0t = node_layer0.transpose()
            song_t = song.transpose()

            sig_der_o = sigmoid_dx(np.sum(dis_net['o'][:-1] * node_layer1t) + dis_net['o'][-1])
            err_der = ((actual_output - target_output) * dis_learning_rate * sig_der_o)  # error derivative and others

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
        rounds += 1

    print()
    print("Starting generator training...")
    target_output = 1  # when the Discriminator thinks the song is real (aka not generated)
    error = total_songs
    song_seeds = [np.array(2000*(r.random() - 0.5)) for k in range(total_songs)]
    rounds = 0
    while error / total_songs > 0.25 and rounds < 24:  # FIXME: change to some value (should research)
        error = 0

        o_weight_gradient = np.zeros((gen_hid_nod1, rag_data_len), dtype=np.float128)
        h1_weight_gradient = np.zeros((gen_hid_nod0, gen_hid_nod1), dtype=np.float128)
        h0_weight_gradient = np.zeros((1, gen_hid_nod0), dtype=np.float128)
        o_bias_gradient = np.zeros(rag_data_len, dtype=np.float128)
        h1_bias_gradient = np.zeros(gen_hid_nod1, dtype=np.float128)
        h0_bias_gradient = np.zeros(gen_hid_nod0, dtype=np.float128)

        output_song_count = 0
        for some_song in song_seeds:
            # Generator vector nodes
            song_vec_h0 = NLC(some_song, gen_net['h0'])
            song_vec_h1 = NLC(song_vec_h0, gen_net['h1'])
            song_vec_o = NLC(song_vec_h1, gen_net['o'])
            song_vector = song_vec_o

            output_songs[output_song_count] = song_vector
            output_song_count += 1

            # Discriminator vector nodes
            node_layer0 = NLC(song_vector, dis_net['h0'])
            node_layer1 = NLC(node_layer0, dis_net['h1'])
            actual_output = NLC(node_layer1, dis_net['o'])[0][0]
            error += (actual_output - target_output) * (actual_output - target_output)

            node_layer0t = node_layer0.transpose()
            node_layer1t = node_layer1.transpose()
            song_t = song_vector.transpose()

            sig_der_o = sigmoid_dx(np.sum(dis_net['o'][:-1] * node_layer1t) + dis_net['o'][-1])

            err_der = (actual_output - target_output) * gen_learning_rate * sig_der_o  # error derivative and terms
            sig_der_h1 = sigmoid_dx(sum(dis_net['h1'][:-1] * node_layer0t) + dis_net['h1'][-1])
            sig_der_h0 = sigmoid_dx(sum(dis_net['h0'][:-1] * song_t) + dis_net['h0'][-1])

            song_h0t = song_vec_h0.transpose()
            song_h1t = song_vec_h1.transpose()

            gen_sig_der_o = sigmoid_dx(sum(gen_net['o'][:-1] * song_h1t) + gen_net['o'][-1])
            temp_o = gen_sig_der_o * np.dot(sig_der_h0 * dis_net['h0'][:-1],
                                            np.dot(sig_der_h1 * dis_net['h1'][:-1],
                                                   err_der * dis_net['o'][:-1])).transpose()

            o_weight_gradient += song_h1t * temp_o
            o_bias_gradient += temp_o[0]

            gen_sig_der_h1 = sigmoid_dx(sum(gen_net['h1'][:-1] * song_h0t) + gen_net['h1'][-1])
            temp_h1 = gen_sig_der_h1 * np.dot(gen_sig_der_o * gen_net['o'][:-1],
                                              np.dot(sig_der_h0 * dis_net['h0'][:-1],
                                                     np.dot(sig_der_h1 * dis_net['h1'][:-1],
                                                            err_der * dis_net['o'][:-1]))).transpose()

            h1_weight_gradient += song_h0t * temp_h1
            h1_bias_gradient += temp_h1[0]

            gen_sig_der_h0 = sigmoid_dx(sum(gen_net['h0'][:-1] * some_song) + gen_net['h0'][-1])
            temp_h0 = gen_sig_der_h0 * np.dot(gen_sig_der_h1 * gen_net['h1'][:-1],
                                              np.dot(gen_sig_der_o * gen_net['o'][:-1],
                                                     np.dot(sig_der_h0 * dis_net['h0'][:-1],
                                                            np.dot(sig_der_h1 * dis_net['h1'][:-1],
                                                                   err_der * dis_net['o'][:-1])))).transpose()
            h0_weight_gradient += some_song * temp_h0
            h0_bias_gradient += temp_h0[0]

        print("Average Gen Error:", error / total_songs)
        gen_net['o'][-1] -= o_bias_gradient / total_songs
        gen_net['h1'][-1] -= h1_bias_gradient / total_songs
        gen_net['h0'][-1] -= h0_bias_gradient / total_songs
        gen_net['o'][:-1] -= o_weight_gradient / total_songs
        gen_net['h1'][:-1] -= h1_weight_gradient / total_songs
        gen_net['h0'][:-1] -= h0_weight_gradient / total_songs
        rounds += 1

    name_counter = 0
    for a_song in output_songs:
        output_song = np.array([np.round(note_range * a_song[0][:note_count_max]),
                               np.round(time_max * a_song[0][note_count_max:])])
        OutputNPArrayToFile(output_song, str(name_counter))
        all_data[total_songs + name_counter] = a_song
        name_counter += 1
