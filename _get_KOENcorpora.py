from pathlib import Path
import numpy as np                     # Import numpy for array manipulation
# import matplotlib.pyplot as plt        # Import matplotlib for charts
# from utils_nb import plot_vectors      # Function to plot vectors (arrows)
from os import walk,path,listdir,makedirs
import pandas as pd
# import openpyxl
from nltk import sent_tokenize,word_tokenize,data
import nltk
from konlpy import jvm
from konlpy.tag import Kkma,Okt

import re

import glob

inputfilepath= '../data/file/corpora/'

keydictionary={"news":"뉴스","culture":"문화", "law":"조례","local_gov":"지자체","guo":"구어","conv":"대화","muno":"문어"}

mycols={"원문": "KO", "번역문": "EN"}
outputfile={"src":"enko_src","tgt":"enko_tgt"}

split_ratio=[99,0.5,0.5]
filter_key="law"
splitname="".join([str(int) for int in split_ratio])+ filter_key if filter_key else "".join([str(int) for int in split_ratio])

# print(splitname)

write_folder= "enko/"+ splitname + "/"


def find_csv_filenames( path_to_dir, suffix=".xlsx", filter_key=filter_key):
    print(f"Read from: {path_to_dir}")
    filenames = listdir(path_to_dir)


    if filter_key is None or filter_key not in list(keydictionary.keys()):
        return [filename for filename in filenames if filename.endswith(suffix)]
    else: # filter results
        print(f"Using key:\t{filter_key}({keydictionary.get(filter_key)})")
        return [filename for filename in filenames if filename.endswith(suffix) and keydictionary.get(filter_key) in filename]

# def get_corpus(filenames):
#     mydataframe = pd.DataFrame(columns=mycols.keys())
#
#     for filename in filenames:
#         print(filename)
#         df = pd.read_excel(path.join(mypath + filename), engine='openpyxl', usecols=mycols.keys())
#         print(len(df))
#
#         mydataframe = mydataframe.append(df, ignore_index=True)
#
#     mydataframe["원문"].to_csv(prefix + outputfile.get("tgt")+".txt", header=False, index=False)
#     mydataframe["번역문"].to_csv(prefix + outputfile.get("src")+".txt", header=False, index=False)
# # get_corpus(filenames)

def get_splitted_corpus(filenames, ratio,padding=True):
    # tokenizer = data.load('tokenizers/punkt/english.pickle')
    # kkma = Kkma()
    # nltk.download('punkt')

    train,valid,test = pd.DataFrame(columns=mycols.keys()),pd.DataFrame(columns=mycols.keys()),pd.DataFrame(columns=mycols.keys())
    splitted_files={}

    # print(filenames[1:2])
    # ko_en, kkma_ko, en_kkma = 0, 0, 0

    for filename in filenames:
        print(filename)
        df = pd.read_excel(path.join(inputfilepath + filename), engine='openpyxl', usecols=mycols.keys())
        print(f"Total: {len(df)}")

        # for index, row in df.iterrows():
        #     print(re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ",row["원문"]))

        #     kkmal,kol,enl=len(kkma.sentences(row['원문'])), len(sent_tokenize(row['원문'])),len(sent_tokenize(row['번역문']))
        #     if kkmal != enl != kol:
        #         print(f"{index} {kkmal}({kol}) {enl} {kkma.sentences(row['원문'])} ({sent_tokenize(row['원문'])})   -->   {sent_tokenize(row['번역문'])}")
        #         if kol != enl: ko_en=ko_en+1
        #         if kol !=kkmal : kkma_ko=kkma_ko+1
        #         if  enl != kkmal: en_kkma=en_kkma+1

        if padding:#padding for punctuation
            df["원문"] = df["원문"].apply(lambda x:  re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", x))
            df["번역문"] = df["번역문"].apply(lambda x: re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", x))

            # re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", input)

        print(df.head(5))

        traindf = df.sample(frac=ratio[0]/sum(ratio),random_state=1)
        train = train.append(traindf, ignore_index=True)
        df=df[~df.index.isin(traindf.index)]
        validdf = df.sample(frac=ratio[1]/(ratio[1]+ratio[2]),random_state=1)
        valid = valid.append(validdf, ignore_index=True)
        testdf = df[~df.index.isin(validdf.index)]
        test = test.append(testdf, ignore_index=True)


        print(f"Train / Valid / Test : {len(traindf)} / {len(validdf)} / {len(testdf)}")
        del(df)

    # print(f"ko_en = {ko_en} \t kkma_en = {en_kkma} \t kkma_ko = {kkma_ko}")

    print(f"Write to: {write_folder}")
    if not path.exists(write_folder):
        makedirs(write_folder)


    train["원문"].to_csv(write_folder + outputfile.get("tgt") + "-train" + ".txt", header=False, index=False, encoding='utf-8')
    train["번역문"].to_csv(write_folder + outputfile.get("src") + "-train" + ".txt", header=False, index=False, encoding='utf-8')
    valid["원문"].to_csv(write_folder + outputfile.get("tgt") + "-valid" + ".txt", header=False, index=False, encoding='utf-8')
    valid["번역문"].to_csv(write_folder + outputfile.get("src") + "-valid" + ".txt", header=False, index=False, encoding='utf-8')
    test["원문"].to_csv(write_folder + outputfile.get("tgt") + "-test" + ".txt", header=False, index=False, encoding='utf-8')
    test["번역문"].to_csv(write_folder + outputfile.get("src") + "-test" + ".txt", header=False, index=False, encoding='utf-8')
    #

# def make_padding(path_to_dir):
#     filenames = listdir(path_to_dir)
#     print(filenames)
#     for file in filenames:
#         print(file[1:])
#         # output = open(file, "w")
#
#         mode = 'a' if path.exists(path_to_dir+file[1:]) else 'w'
#         output=open(path_to_dir+file[1:], mode)
#
#         input = open(file,"r+")
#
#         for line in input:
#             output.write(re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", line))
#
#         input.close()
#         output.close()


# make_padding(prefix)


# mydataframe=pd.read_csv(outputfile,index_col=0)
# print(len(mydataframe))
# mydataframe.rename(columns=mycols,inplace=True)
# mydataframe.to_csv("_"+outputfile)

# Tokenizing
#
# jvm.init_jvm("C:\Program Files\Java\jdk1.8.0_221\jre\\bin\server\jvm.dll")  # TODO to config file
# kkma = Kkma()
# okt=Okt() #fix error, the Okt gives the most appropriate results
#
#
# mydataframe['KO1'] = mydataframe['KO']
#
# for index, row in mydataframe.head(5).iterrows():
#     mydataframe.at[index, "EN"] = word_tokenize(mydataframe.at[index, "EN"])
#     mydataframe.at[index, "KO"] = kkma.morphs(mydataframe.at[index, "KO"])
#     mydataframe.at[index, "KO1"] = okt.morphs(mydataframe.at[index, "KO1"]) #error

# print(mydataframe)

def create_yaml():
    import yaml

    data={"save_data":write_folder+"run/example",
          "src_vocab": write_folder+"run/enko_example.vocab.src",
          "tgt_vocab": write_folder+"run/enko_example.vocab.tgt",
          "overwrite": True,
          "data":
              {
                  "corpus_1":{"path_src": write_folder+"enko_src-train.txt","path_tgt": write_folder+"enko_tgt-train.txt"},
                  "valid":{"path_src": write_folder+"enko_src-valid.txt","path_tgt": write_folder+"enko_tgt-valid.txt"}
              },
          "save_model":write_folder+"run/enko_model",
          "save_checkpoint_steps": 500,
          "train_steps": 1000,
          "valid_steps": 500

          # "world_size": 1
          # "gpu_ranks": [0]
          }


    with open("_".join(list(mycols.values()))+splitname+'.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    print("Configuration file "+ "_".join(list(mycols.values()))+splitname+'.yaml' + " is created.")

filenames = find_csv_filenames(inputfilepath)
print(filenames)

get_splitted_corpus(filenames, split_ratio)
create_yaml()