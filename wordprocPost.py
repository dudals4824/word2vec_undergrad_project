#! /usr/bin/env python
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
import pandas as pd

validationdf = pd.read_csv("./data/test_for_eval_50k_1000.csv")
model = Word2Vec.load("./data/new_50kwvModel")
wv = model.wv
inverse_dict = pd.read_csv("./data/inverse_dictionary.csv", dtype='str', index_col=0)
keys_list = inverse_dict.index.values
learning_rate = 0.001
batch_size = 256 #256

def batch_validation(validationdf, start_index):
    x_dev = []
    broken_word_list = list()
    target_word_list = list()
    data = validationdf.values
    data_size = len(validationdf)
    cur_idx = start_index

    while len(x_dev) < batch_size:
        input_x = [int(x) for x in data[cur_idx][0].split(" ")[0:10]]
        broken_word = data[cur_idx][0].split(" ")[10]
        target_word = data[cur_idx][1]
        x_dev.append(input_x)
        broken_word_list.append(broken_word)
        target_word_list.append(target_word)
        cur_idx += 1
        if cur_idx == data_size:
            break
    start_index = cur_idx
    return x_dev, broken_word_list, target_word_list, start_index

checkpoint_file = tf.train.latest_checkpoint("./rnn_runs/50_2nd_sub_checkpoint/")
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        # Tensors we want to evaluate
        # top_k : k== 100
        predictions = graph.get_operation_by_name("topk").outputs[1]
        start_idx = 0
        all_data_len = len(validationdf)
        correct_len = 0

        batch_count = 0
        batch_num = int(all_data_len/batch_size)

        while True:
            # print(str(batch_count) +" / "+ str(batch_num))
            x_dev, broken_word_list, target_word_list, start_idx = batch_validation(validationdf, start_idx)
            if len(x_dev) == 0:
                break
            batch_predictions = sess.run(predictions, {input_x: x_dev})
            # batch_predictons의 원소 하나는 100개의 index를 갖고있는 배열이다.
            for iter in range(0, len(x_dev)):
                # broken word list는 불러왔음
                # batch_predictons를 이용해서 후보 단어 리스트를 만든다.
                # expansion_cand_list : table에 있는 단어를 확인하고 그것의 파생단어를 list에 넣는다.
                expansion_cand_list = list()
                for i in range(0, 100):
                    std_word = wv.index2word[int(batch_predictions[i])]
                    if std_word in keys_list:
                        expansion_str = inverse_dict.loc[std_word].values[0].split(" ")
                        for s in expansion_str:
                            expansion_cand_list.append(s)

                broken_word = broken_word_list[iter]
                broken_word_len = len(broken_word)
                scoring_list = list()
                for i in range(0, len(expansion_cand_list)):
                    # 길이에 대한 점수
                    # Length difference = 100 * (Length difference between two words) / (Length of target word)
                    cand_word_len = len(expansion_cand_list[i])
                    length_difference = 100 * abs(broken_word_len - cand_word_len) / broken_word_len
                    # 앞뒤 철자가 얼마나 같은지
                    # Matching Point = 100*(Length of Longest prefix match + Length of keyword)/(Longer length of two Word)
                    spel_target_word = broken_word.split("")
                    spel_cand_word = expansion_cand_list[i].split("")
                    prefix_match = 0
                    suffix_match = 0
                    # 앞글자 같았다 안같았다 확인하는 bool 객체
                    # prefix, suffix 길이 찾는거 module화 하기
                    spel_check = False
                    matching_point = 0
                    if broken_word_len > cand_word_len:
                        # 앞 글자 확인
                        if spel_target_word[0] == spel_cand_word[0]:
                            prefix_match += 1
                            spel_check = True

                        if spel_check == True:
                            for i in range(1, cand_word_len):
                                if spel_target_word[i] == spel_cand_word[i]:
                                    prefix_match += 1
                                else:
                                    break

                        # spel_check 초기화
                        spel_check = False
                        # 뒷글자 확인하기
                        if spel_target_word[broken_word_len - 1] == spel_cand_word[cand_word_len - 1]:
                            suffix_match += 1
                            spel_check = True

                        if spel_check == True:
                            for i in range(cand_word_len - 2, -1):
                                if spel_target_word[i] == spel_cand_word[i]:
                                    suffix_match += 1
                                else:
                                    break

                        matching_point = 100 * (prefix_match + suffix_match) / broken_word_len

                    elif broken_word_len < cand_word_len:
                        # 앞 글자 확인
                        if spel_target_word[0] == spel_cand_word[0]:
                            prefix_match += 1
                            spel_check = True

                        if spel_check == True:
                            for i in range(1, broken_word_len):
                                if spel_target_word[i] == spel_cand_word[i]:
                                    prefix_match += 1
                                else:
                                    break

                        # spel_check 초기화
                        spel_check = False
                        # 뒷글자 확인하기
                        if spel_target_word[broken_word_len - 1] == spel_cand_word[cand_word_len - 1]:
                            suffix_match += 1
                            spel_check = True

                        if spel_check == True:
                            for i in range(broken_word_len - 2, -1):
                                if spel_target_word[i] == spel_cand_word[i]:
                                    suffix_match += 1
                                else:
                                    break
                        matching_point = 100 * (prefix_match + suffix_match) / cand_word_len
                    else:
                        # 앞 글자 확인
                        if spel_target_word[0] == spel_cand_word[0]:
                            prefix_match += 1
                            spel_check = True

                        if spel_check == True:
                            for i in range(1, cand_word_len):
                                if spel_target_word[i] == spel_cand_word[i]:
                                    prefix_match += 1
                                else:
                                    break

                        # spel_check 초기화
                        spel_check = False
                        # 뒷글자 확인하기
                        if spel_target_word[broken_word_len - 1] == spel_cand_word[cand_word_len - 1]:
                            suffix_match += 1
                            spel_check = True

                        if spel_check == True:
                            for i in range(cand_word_len - 2, -1):
                                if spel_target_word[i] == spel_cand_word[i]:
                                    suffix_match += 1
                                else:
                                    break

                        matching_point = 100 * (prefix_match + suffix_match) / broken_word_len

                    integrated_point = 100 * (matching_point - length_difference) / 2
                    scoring_list.append(integrated_point)

                highest_score_idx = int(np.argmax(scoring_list))
                highest_score_word = expansion_cand_list[highest_score_idx]

                if highest_score_word == target_word_list[iter]:
                    correct_len += 1
            batch_count += 1
            print("지금까지 맞춘 개수: " + str(correct_len / batch_size*batch_count))
            if batch_count > batch_num:
                break

