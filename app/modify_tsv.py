import csv 

pre_label = 0
pre_html_id = 0
tmp_sentence = ''


# 学習用データ加工
with open('./dataset/training.tsv', 'r') as f_in, open('./processed_dataset/modified_training.tsv', 'w') as f_out:

    for row in f_in:
        sid, sentence, html_id, label = row.strip().split('\t')

        if label == 0:     
            f_out.write('{}\t{}\t{}\t{}\n'.format(sid, sentence, html_id, label ))
            pre_label=label
        elif label == pre_label and html_id == pre_html_id:
            tmp_sentence += ' {}'.format(sentence)

        else:
            f_out.write('{}\t{}\t{}\t{}\n'.format(sid, tmp_sentence, html_id, label ))
            tmp_sentence = ''
            tmp_sentence += ' {}'.format(sentence)
            pre_label=label
            pre_html_id=html_id


# 学習用データ整形
with open('./processed_dataset/modified_training.tsv', 'r') as f_in, open('./processed_dataset/train_ft.txt', 'w') as f_out:
    for row in f_in:
        sid, sentence, html_id, label = row.strip().split('\t')
        f_out.write('__label__{} {}\n'.format(label, sentence))