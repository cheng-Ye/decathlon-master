
import os
import json

path='srl//wiki1.train.qa.txt'

#'srl'
with open(os.path.expanduser(path)) as f:
            for line in f:

                print(line)
                ex = json.loads(line)
                t = ex['type']
                aa = ex['all_answers']
                context, question, answer = ex['context'], ex['question'], ex['answer']
                context_question = get_context_question(context, question) 
                ex = data.Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields)
                examples.append(ex)
                ex.squad_id = len(all_answers)
                all_answers.append(aa)
                if subsample is not None and len(examples) >= subsample: 
                    break



