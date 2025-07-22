import os


def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):  # relaions.dict和entities.dict中的id都是按顺序排列的
            rel, id = line.strip().split("\t")
            index[rel] = id
            rev_index[id] = rel
    return index, rev_index


entity2id, id2entity = load_index(os.path.join('entity2id.txt'))

word_list = set()  # todo 无序且不重复
for entity_str in entity2id.keys():
    # todo 如果有括号，则括号前和括号内的内容分别加入到 word_list 中；否则，内容直接加入到 word_list 中
    if "(" in entity_str and ")" in entity_str:
        # todo 找出小括号的索引
        begin = entity_str.find('(')
        end = entity_str.find(')')
        w1 = entity_str[:begin].strip()  # todo 括号前的内容
        w2 = entity_str[begin + 1: end]  # todo 括号内的内容
        word_list.add(w1)
        word_list.add(w2)
    else:
        word_list.add(entity_str)

word2id = {word: id for id, word in enumerate(word_list)}

eid2wid = []
for id in range(len(id2entity.keys())):
    entity_str = id2entity[str(id)]
    if "(" in entity_str and ")" in entity_str:
        # todo 找出小括号的索引
        begin = entity_str.find('(')
        end = entity_str.find(')')
        w1 = entity_str[:begin].strip()  # todo 括号前的内容
        w2 = entity_str[begin + 1: end]  # todo 括号内的内容
        eid2wid.append([str(entity2id[entity_str]), "0", str(word2id[w1])])  # isA关系
        eid2wid.append([str(entity2id[entity_str]), "1", str(word2id[w2])])  # 隶属关系
    else:
        eid2wid.append([str(entity2id[entity_str]), "2", str(word2id[entity_str])])

with open("e-w-graph.txt", "w") as f:
    for line in eid2wid:
        f.write("\t".join(line) + '\n')
