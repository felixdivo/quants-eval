import numpy as np
import pandas as pd
import opp_qa.embedding as ebd
import operator
import sys
import scipy as sc
from collections import defaultdict
from nltk import word_tokenize

# from keras.preprocessing.sequence import pad_sequences
from torch.nn.utils.rnn import pad_sequence
import pickle
from datasets import load_dataset, VerificationMode
import datasets

from tqdm import tqdm


def int_to_answers(source_data, k=100):
    # #     data_path = 'data/train_qa'
    #     df = pd.read_pickle(data_path)
    # #     df = df[(df.question_family_index!=14) & (df.question_family_index!=15)] #14/15questions already filtered out!
    #     answers = df[['answer']].values.tolist()
    #     freq = defaultdict(int)
    #     for answer in answers:
    #         freq[answer[0].lower()] += 1
    #     int_to_answer = sorted(freq.items(),key=operator.itemgetter(1),reverse=True)[0:k] # top k answers
    #     int_to_answer = [answer[0] for answer in int_to_answer]

    # fix answer coding stype: consistent with s123 coding!
    if source_data == "opp":
        int_to_answer = [
            "no",
            "yes",
            "0",
            "1",
            "2",
            "open the front door",
            "clean the table",
            "open the third drawer",
            "close the front door",
            "toggle the switch",
            "close the third drawer",
            "open the second drawer",
            "close the first drawer",
            "close the second drawer",
            "open the first drawer",
            "close the back door",
            "open the back door",
            "close the fridge",
            "open the fridge",
            "close the dishwasher",
            "drink from the cup",
            "open the dishwasher",
            "3",
            "6",
            "4",
            "7",
        ]

        # good fixed coding...
    #     int_to_answer = ['no','yes',
    #                      '0', '1', '2','3', '4', '5', '6', '7',
    #                      'open the front door', 'close the front door',
    #                      'open the first drawer', 'close the first drawer',
    #                      'open the second drawer', 'close the second drawer',
    #                      'open the third drawer', 'close the third drawer',
    #                      'open the back door', 'close the back door',
    #                      'open the fridge', 'close the fridge',
    #                      'open the dishwasher', 'close the dishwasher',
    #                      'toggle the switch', 'drink from the cup', 'clean the table'
    #                     ]
    else:  # for ES dataset
        int_to_answer = [
            "yes",
            "no",
            "2",
            "3+",
            "0",
            "1",
            "3",
            "walk",
            "sit down",
            "stand up",
            "ride bicycle",
            "lie down",
            "run",
        ]

    return int_to_answer


# train_data_path = '../SQA_data_gen/generated_sqa_data/sqa_S1_1800_900.pkl'
# top_answers = int_to_answers(train_data_path)


def answers_to_onehot(train_data_path, source_data):
    top_answers = int_to_answers(train_data_path, source_data)
    answer_number = len(top_answers)
    answer_to_onehot = {}
    for i, word in enumerate(top_answers):
        onehot = np.zeros(answer_number + 1)
        onehot[i] = 1.0
        answer_to_onehot[word] = onehot
    return answer_to_onehot


# answer_to_onehot_dict = answers_to_onehot()


def load_data(splits=["val", "train"]):
    ds = load_dataset(
        "dasyd/OppQA-500",
        "binary",
        data_dir="binary",
        data_files={split: f"{split}-*" for split in splits},
        verification_mode=VerificationMode.NO_CHECKS,
        num_proc=len(splits),
    )
    return ds


def get_answers_matrix(data_path, source_data, max_ans_num=100):
    # 	if split == 'train':
    # 		data_path = 'data/train_qa'
    # 	elif split == 'val':
    # 		data_path = 'data/val_qa'
    # 	else:
    # 		print('Invalid split!')
    # 		sys.exit()
    # #     data_path = 'data/train_qa'
    top_answers = int_to_answers("opp")
    answer_number = len(top_answers)
    answer_to_onehot_dict = answers_to_onehot(data_path, source_data)

    df = pd.read_pickle(data_path)
    answers = df[["answer"]].values.tolist()
    answer_matrix = np.zeros((len(answers), answer_number + 1))
    default_onehot = np.zeros(answer_number + 1)
    default_onehot[answer_number] = 1.0

    for i, answer in enumerate(tqdm(answers)):
        answer_matrix[i] = answer_to_onehot_dict.get(answer[0].lower(), default_onehot)

    return answer_matrix


def get_questions_matrix():
   
    word_idx = ebd.load_idx()
    ds = load_data(["train"])

    train_ds = ds["train"]

  
    int_to_answer = [
            "no",
            "yes",
            "0",
            "1",
            "2",
            "open the front door",
            "clean the table",
            "open the third drawer",
            "close the front door",
            "toggle the switch",
            "close the third drawer",
            "open the second drawer",
            "close the first drawer",
            "close the second drawer",
            "open the first drawer",
            "close the back door",
            "open the back door",
            "close the fridge",
            "open the fridge",
            "close the dishwasher",
            "drink from the cup",
            "open the dishwasher",
            "3",
            "6",
            "4",
            "7",
        ]


    def tokenize_and_pad(examples, word_idx, maxlen=31):
        # Tokenize the question
        tokens = word_tokenize(examples["question"])
        
        # Convert tokens to their corresponding indices
        tok_indices = [word_idx.get(token.lower(), 0) for token in tokens]
        
        # Pad the token indices
        padded_tok = tok_indices[:maxlen] + [0] * max(0, maxlen - len(tok_indices))
        
        return {'text_idx': padded_tok, "answer_text": int_to_answer[examples["answer"]]}
    
    # Apply the transformation to the dataset
    tok_data = train_ds.map(lambda x: tokenize_and_pad(x, word_idx=word_idx), batched=False)
    

    return tok_data.with_format("torch")


    # # questions = train_ds.map(lambda x: list(map(lambda word: word_idx.get(word.lower(),0), word_tokenize(x["question"]))),batched=True)

    # # map(lambda word: word_idx.get(word.lower(),0), word_tokenize(ds["train"][0]["question"]))

    
def get_sensory_context(
    train_data_path, context_data_path, source_data, win_len, embedding=False
):

    # load sensory context dictionary
    with open(context_data_path, "rb") as handle:
        sensory_data = pickle.load(handle)

    # unique key in the sensory context
    unique_key = list(sensory_data["raw"].keys())

    # load pickle question data
    question_data = pd.read_pickle(train_data_path)

    # store context key of question data into a numpy array
    context_key_list = (
        question_data["context_source_file"]
        + "_"
        + question_data["context_start_point"].astype(str)
    )
    context_key_list = np.array(context_key_list)

    # if using sensory embedding
    if embedding:
        sensory_matrix = np.zeros((len(question_data), 128))

        for key_i in tqdm(unique_key):
            key_index = np.where(context_key_list == key_i)
            sensory_matrix[key_index, :] = sensory_data["embedding"][key_i][0, :]

    #  if using raw data
    else:
        if source_data == "opp":
            sensory_matrix = np.zeros(
                (len(question_data), win_len, 77), dtype="float32"
            )
        else:
            sensory_matrix = np.zeros(
                (len(question_data), win_len, 225), dtype="float32"
            )

        for key_i in tqdm(unique_key):
            key_index = np.where(context_key_list == key_i)
            sensory_matrix[key_index, :] = sensory_data["raw"][key_i]

    return sensory_matrix
