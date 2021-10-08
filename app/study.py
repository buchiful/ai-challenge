import fasttext as ft


# 学習用データ整形
with open('training.tsv', 'r') as f_in, open('train_ft.txt', 'w') as f_out:
    for row in f_in:
        text, label = row.strip().split('\t')
        f_out.write('__label__{} {}\n'.format(label, text))

# テストデータ整形
with open('test.tsv', 'r') as f_in, open('test_ft.txt', 'w') as f_out:
    for row in f_in:
        text, label = row.strip().split('\t')
        f_out.write('__label__{} {}\n'.format(label, text))

# 学習:モデル作成
model = ft.train_supervised(input="train_ft.txt")
model.save_model("model.bin")”


# 推論テスト
ret = model.predict("Environmental laws may require the Company to incur substantial expenses and may materially reduce the affected property’s value or limit the Company’s ability to use or sell the affected property.	Form10k_09")
print(ret)


# 推論テスト
ret = model.test("test_ft.txt")
