import pandas as pd  
import re
import os 
import pickle
from tqdm import tqdm

getcwd=os.path.join(os.getcwd(),'csv')
csvlist=os.listdir(getcwd)
word=[]
print(csvlist)
for csv in csvlist:
    print('current_csv',csv)
    current_csv=pd.read_csv(os.path.join(getcwd,csv),index_col=0,nrows=1000)
    for context,question,answer in tqdm(zip(current_csv['context'],current_csv['quetion'],current_csv['answer'])):
        context=context+question+answer
        filtered_English_str = [ i.lower().strip()  for i in re.findall('[A-Za-z0-9 ]+', context)  if i.lower().strip() !='']   #保留字母
        #print(filtered_English_str)
        for i in filtered_English_str :
            if re.findall(' ',i):
                word.extend( [i.lower().strip()  for i in i.split(' ') if i.lower().strip()!=''  ])
            else:
                word.append(i.lower().strip())
        word=list(set(word))

word.sort()
voc = dict(enumerate(word))
print('has  words:',len(voc))
#print(voc)
with open('word_dict1.pkl','wb')as f:
    pickle.dump(voc,f)