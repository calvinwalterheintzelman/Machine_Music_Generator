import os
import numpy as np
import numpy.random as r
np.seterr(all='raise')
r.seed(4685)


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


for midi_name in os.listdir('/home/calvin/Desktop/machine_music/newmidis'):
    os.system('./midicsv-1.1/midicsv newmidis/' + midi_name + " csvrags/" + midi_name[:len(midi_name) - 4] + '_csv.txt')

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
checker_thing = []
for file_name in os.listdir('/home/calvin/Desktop/machine_music/csvrags'):
    print(file_name)
    checker_thing.append(file_name.split("_")[0])
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

    '''
    # Add transpoitions of each music piece to data
    for k in range(5):
        all_data.append(np.array([temp_data[0] - k - 1, temp_data[1]]))
    for k in range(6):
        all_data.append(np.array([temp_data[0] + k + 1, temp_data[1]]))
    '''


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

    end_time = None  # assigned here to be used later
    note_length = 60  # can change to be anything, really. The length of each note played
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

    os.system('./midicsv-1.1/csvmidi outputcsv/' + name + '_csv.txt test_new_midis/' + name + ".mid")


for j in range(len(all_data)):
    OutputNPArrayToFile(all_data[j], str(j))

csodf = 0
for thing in checker_thing:
    os.system("diff newmidis/" + thing + ".mid test_new_midis/" + str(csodf) + '.mid')
    csodf += 1
