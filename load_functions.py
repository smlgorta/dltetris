import numpy as np
import pandas as pd
import ast
import pickle

def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

def pickleSave(path, obj):
    fileObject = open(path, 'wb')
    pickle.dump(obj, fileObject)
    fileObject.close()
    return;

def pickleLoad(path):
    fileObject = open(path, 'rb')
    return pickle.load(fileObject)


def loadAndParseData(input_path, output_path):
    print("loading csv")
    csv = pd.read_csv(input_path)
    def parseStates(states):
        parsed_states = []
        for i in range(states.shape[0]):
            parsed_states.append(parseState(states[i]).flatten())
        return parsed_states

    def parseState(state):
        splitted_state = state.split("|")
        board = np.zeros((10, 10))
        for row in range(1, len(splitted_state)):
            splitted_row = splitted_state[row].split(":")
            for col in range(0, len(splitted_row)):
                if splitted_row[col] != "":
                    board[row-1][int(splitted_row[col])] = 1
        return board

    print("parsing states")
    csv["stateaction"] = parseStates(csv["stateaction"])
    csv = pd.DataFrame(csv)
    # print("writing to csv")
    # csv.to_csv(output_path, header=True, index=False)
    print("writing to pickle")
    fileObject = open(output_path, 'wb')
    pickle.dump(csv, fileObject)
    fileObject.close()

def loadParsedReshapeAndFlatten(input_path, input_outpath, output_outpath):
    # print("loading parsed states from data/parsed.csv")
    # csv = pd.read_csv(input_path, converters={'stateaction':from_np_array})

    print("reading from pickle")
    csv = pickleLoad(input_path)

    print("grouping by state")
    grouped_csv = csv.groupby("step", as_index=True)
    print("setting indices for actions")
    indices = []
    for i, row in grouped_csv.size().iteritems():
        indices.extend(grouped_csv.get_group(i).reset_index().index.values)
    print("attaching indices to main frame")
    csv['option'] = indices
    print("reshaping")
    pivoted_csv = csv.pivot(index="step", columns='option')

    def softmax(x):
        z = np.sum(np.exp(x))
        sm = np.exp(x) / z
        return np.nan_to_num(sm);

    input_ret = []
    output_ret = []
    for index, row in pivoted_csv.iterrows():
        input_row = []
        repeat = 0
        for i in range(34):
            if row.get_values()[i] is not None:
                input_row = np.append(input_row, row.get_values()[i])
            else:
                if row.get_values()[repeat] is None:
                    repeat = 0
                input_row = np.append(input_row, row.get_values()[repeat])
                repeat += 1

        # print("row: ", input_row)
        input_ret.append(input_row)
        output_row = []
        repeat = 0
        for i in range(34, 68):
            if row.get_values()[i] is not None:
                output_row = np.append(output_row, row.get_values()[i])
            else:
                if row.get_values()[34 + repeat] is None:
                    repeat = 0
                output_row = np.append(output_row, row.get_values()[34+repeat])
                repeat += 1
        output_ret.append(softmax(output_row.flatten()))
    output_ret = np.array(output_ret)
    input_ret = np.array(input_ret)
    print("writing to pickle")
    print(input_ret.shape)
    print(output_ret.shape)
    pickleSave(output_outpath, output_ret)
    pickleSave(input_outpath, input_ret)

def loadParsedAndMakePairwiseComparisons(input_path, output_path):
    l = pickleLoad(input_path)
    steps = l["step"]
    steps = steps.drop_duplicates()
    steps = steps.reset_index(drop = True)
    pw_dataset = []
    for i in range(len(steps)):
        step = l[l['step'] == steps[i]]
        for j in range(len(step)):
            for k in range(len(step)):
                pw = []
                if(j != k):
                    pw = [l['stateaction'][j],l['stateaction'][k],l['value'][j]-l['value'][k]]
                    if(len(pw_dataset) == 0):
                        pw_dataset = pw
                    else:
                        pw_dataset = np.vstack([pw_dataset, pw])
    pickleSave(output_path, pw_dataset)