import json

def return_novel_label_names():
# 打开 JSON 文件
    with open('imagenet_class_index.json', 'r', encoding='utf-8') as file:
        # 加载 JSON 数据
        data = json.load(file)

    ids=["n01930112","n01981276","n02099601","n02110063","n02110341","n02116738","n02129165","n02219486","n02443484","n02871525","n03127925","n03146219","n03272010","n03544143","n03775546","n04146614","n04149813","n04418357","n04522168","n07613480"]
    label_names = []
    for id in ids:
        for i in data.keys():
            if data[i][0] == id:
                label_names.append(data[i][1])
    
    return label_names