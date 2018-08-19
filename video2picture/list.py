"""
生成批量重命名的文件名的.txt文档
"""
name = []
with open('./URL.txt', 'r', encoding='utf-8') as fo:
    for line in fo:
        line = line.strip().split('/')
        name.append(line[-1])

f = open('./list.txt', 'a')
for i in range(200):
    f.write('ren' + ' ' + name[i] + ' ' + str(i+1) + '.mp4')
    f.write('\n')
f.close()