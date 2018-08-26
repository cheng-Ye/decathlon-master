# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import torch
import docx
import re
import os 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class Wordlist(object):
    def __init__(self, filename):

        word=[]
        for file in filelist:
            file=docx.Document(os.path.join(getcwd,file))
                    #self.len = xy.shape[0]
                    #self.x_data = torch.from_numpy(xy[:, 0:-1])#关于np的数组操作，http://blog.csdn.net/liangzuojiayi/article/details/51534164
                    #self.y_data = torch.from_numpy(xy[:, [-1]])
            #输出每一段的内容
            for para in file.paragraphs[:10]:
                filtrate_nonChinese = re.compile(u'[^\u4E00-\u9FA50-9]')#非中文   

                filtered_English_str = [ i.lower().strip()  for i in re.findall('[A-Za-z ]+', para.text)  if i.lower().strip() !='']   #保留字母
                for i in filtered_English_str:
                    if re.findall(' ',i):
                        filtered_English_str.remove(i)
                        filtered_English_str.extend( i.split(' '))

                filtered_Chinese = filtrate_nonChinese.sub(r' ', para.text)#replace
                filtered_Chinese_str=[i for i in ' '.join(filtered_Chinese.replace(' ','')) if i!=' ' ]
                filtered_Chinese_str.extend(filtered_English_str)
                word.extend(filtered_Chinese_str)
                word=list(set(word))     
        word.sort()
        self.size = len(word)
        self.word=word

        self.voc = dict(enumerate(self.word))

        self.reverse_voc = {v:k for k,v in self.voc.items()}

    def getID(self, word):
        try:
            return self.reverse_voc[word]
        except:
            return -1

    def  getWord(self,wordid):

        return self.voc[wordid]


if __name__ == '__main__':
    
    getcwd=os.path.join(os.getcwd(),'word')
    filelist=os.listdir(getcwd)

    wordlist=Wordlist(filelist)

    print(wordlist.voc)

    print(wordlist.getID('高'),wordlist.getWord(225))
