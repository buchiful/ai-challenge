# -*- coding: utf-8 -*-
import fasttext as ft


# 学習用データ整形
with open('./dataset/training.tsv', 'r') as f_in, open('./processed_dataset/train_ft.txt', 'w') as f_out:

    for row in f_in:
        sid, sentence, html_id, label = row.strip().split('\t')
        f_out.write('__label__{} {}\n'.format(label, sentence))

# 学習:モデル作成
model = ft.train_supervised("./processed_dataset/train_ft.txt", label_prefix='__label__',epoch=1000, loss="softmax")
model.save_model("./models/modified_model.bin")