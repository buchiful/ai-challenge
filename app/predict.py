# -*- coding: utf-8 -*-
import fasttext as ft

# modelの設定
model = ft.load_model("./models/modified_model.bin")


# 推論テスト
print("推論テスト")
ret = model.predict("Environmental laws may require the Company to incur substantial expenses and may materially reduce the affected property’s value or limit the Company’s ability to use or sell the affected property.")
print(ret)

# 推論
with open('./dataset/test.tsv', 'r') as f_in, open('./results/test_output2.tsv', 'w') as f_out:
    for row in f_in:
        sid, sentence, html_id = row.strip().split('\t')
        ret = model.predict(sentence)
        label = ret[0][0][9]
        f_out.write('{}\t{}\t{}\t{}\n'.format(sid, sentence, html_id, label))


# アウトプット整形
with open('./results/test_output2.tsv','r') as f_in, open('./results/submission2.tsv', 'w') as f_out:
    for row in f_in:
        sid, sentence, html_id, label = row.strip().split('\t')
        f_out.write('{}\t{}\n'.format(sid, label))