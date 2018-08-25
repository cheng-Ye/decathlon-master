import os
import math
import numpy as np
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .common import positional_encodings_like, INF, EPSILON, TransformerEncoder, TransformerDecoder, PackedLSTM, LSTMDecoderAttention, LSTMDecoder, Embedding, Feedforward, mask


class MultitaskQuestionAnsweringNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        #self.field = field
        #self.args = args
        self.pad_idx = 1
        

        vocab_size=12628     #字典大小



        dimension=200
        dropout_ratio=0.2

        self.encoder_embeddings = Embedding(vocab_size, dimension, 
            dropout=dropout_ratio)
        self.decoder_embeddings = Embedding(vocab_size, dimension, 
            dropout=dropout_ratio)
     
        self.bilstm_before_coattention = PackedLSTM(dimension, dimension,
            batch_first=True, dropout=dropout_ratio, bidirectional=True, num_layers=1)
        self.coattention = CoattentiveLayer(dimension, dropout=dropout_ratio)
        dim = 2*dimension + dimension + dimension

        self.context_bilstm_after_coattention = PackedLSTM(dim, dimension,
            batch_first=True, dropout=dropout_ratio, bidirectional=True, 
            num_layers=1)
        self.self_attentive_encoder_context = TransformerEncoder(dimension, 3, 150, 2, dropout_ratio)
        self.bilstm_context = PackedLSTM(dimension, dimension,
            batch_first=True, dropout=dropout_ratio, bidirectional=True, 
            num_layers=1)

        self.question_bilstm_after_coattention = PackedLSTM(dim, dimension,
            batch_first=True, dropout=dropout_ratio, bidirectional=True, 
            num_layers=1)
        self.self_attentive_encoder_question = TransformerEncoder(dimension, 3, 150, 2, dropout_ratio)
        self.bilstm_question = PackedLSTM(dimension, dimension,
            batch_first=True, dropout=dropout_ratio, bidirectional=True, 
            num_layers=1)

        self.self_attentive_decoder = TransformerDecoder(dimension, 3, 150, 2, dropout_ratio)
        self.dual_ptr_rnn_decoder = DualPtrRNNDecoder(dimension, dimension,
            dropout=dropout_ratio, num_layers=1)

        self.generative_vocab_size = min(vocab_size, 50000)
        self.out = nn.Linear(dimension, self.generative_vocab_size)

        self.dropout = nn.Dropout(0.4)
        with open('word_dict1.pkl','rb')as f:
            voc=pickle.load(f)
        self.voc=voc
        self.reverse_voc = {v:k for k,v in self.voc.items()}

    def set_embeddings(self, embeddings):
        self.encoder_embeddings.set_embeddings(embeddings)
        self.decoder_embeddings.set_embeddings(embeddings)

    def genBatch(self,data):
        #maxsentencenum = len(data[0])
        maxsentencenum = max([len(item) for item in data])
        for doc in data:
            for i in range(maxsentencenum - len(doc)):
                doc.append(1)
        return data 
            

    def forward(self, batch):
    
        #context, context_lengths, context_limited    = batch['context'],  batch['context_lengths'],  batch['context_limited']
        #question, question_lengths, question_limited = batch['question'], batch['question_lengths'], batch['question_limited']
    
        context ,question,answer= batch['context'],batch['quetion'],batch['answer']
        context_indices=[];question_indices=[];answer_indices=[]
        import re 
        for context,question,answer in zip(context,question,answer):
            filtered_English_str = [ i.lower().strip()  for i in re.findall('[A-Za-z0-9 ]+', context)  if i.lower().strip() !='']   #保留字母
            word=[]
            for i in filtered_English_str :
                if re.findall(' ',i):
                    word.extend( [i  for i in i.split(' ') if i!=''  ])
                else:
                    word.append(i.lower().strip())

            context_indices.append([self.reverse_voc[word] for word in word])

            filtered_English_str = [ i.lower().strip()  for i in re.findall('[A-Za-z0-9 ]+', question)  if i.lower().strip() !='']   #保留字母
            word=[]
            for i in filtered_English_str :
                if re.findall(' ',i):
                    word.extend( [i  for i in i.split(' ') if i!=''  ])
                else:
                    word.append(i.lower().strip())

            question_indices.append([self.reverse_voc[word] for word in word])

            filtered_English_str = [ i.lower().strip()  for i in re.findall('[A-Za-z0-9 ]+', answer)  if i.lower().strip() !='']   #保留字母
            word=[]
            for i in filtered_English_str :
                if re.findall(' ',i):
                    word.extend( [i  for i in i.split(' ') if i!=''  ])
                else:
                    word.append(i.lower().strip())

            answer_indices.append([self.reverse_voc[word] for word in word])

        #context_indices=torch.LongTensor(torch.stack(context_indices, 0))
        context_lengths=[len(i) for i in context_indices]
        context_indices=Variable(torch.LongTensor(self.genBatch(context_indices)))
        context=context_indices
        #print(context_lengths,context)
        #print(context_indices[:2])
    
        #print(question_indices)
        #question_indices=torch.LongTensor(torch.stack(question_indices, 0))
        question_lengths=[len(i) for i in question_indices]
        question_indices=Variable(torch.LongTensor(self.genBatch(question_indices)))
        #print(question_lengths,question)

        question= question_indices
        #question, question_lengths, question_limited = batch['question'], batch['question_lengths'], batch['question_limited']
        
        if self.training:
            print('trainng')
            answer_lengths=[len(i) for i in answer_indices]
            answer_indices=Variable(torch.LongTensor(self.genBatch(answer_indices)))
            answer=answer_indices
            #limited_idx_to_full_idx  =batch['limited_idx_to_full_idx']
            #print( torch.unsqueeze(answer[:12], 0))
        else:
            print('test')
            answer_limited = batch['answer_limited']
            limited_idx_to_full_idx  =batch['limited_idx_to_full_idx ']
        oov_to_limited_idx={}
        pad_idx = 1

        #print(context,context_lengths,answer)

        '''
        def map_to_full(x):
            return limited_idx_to_full_idx[x]
        self.map_to_full = map_to_full
        '''
        context_embedded = self.encoder_embeddings(context)

        #print(context,context_embedded)
        #print('============================',question)
        question_embedded = self.encoder_embeddings(question)

        context_encoded = self.bilstm_before_coattention(context_embedded, context_lengths)[0]
        question_encoded = self.bilstm_before_coattention(question_embedded, question_lengths)[0]

        context_padding = context.data == self.pad_idx
        question_padding = question.data == self.pad_idx

        coattended_context, coattended_question = self.coattention(context_encoded, question_encoded, context_padding, question_padding)

#        context_summary = self.dropout(torch.cat([coattended_context, context_encoded, context_embedded], -1))
        context_summary = torch.cat([coattended_context, context_encoded, context_embedded], -1)
        condensed_context, _ = self.context_bilstm_after_coattention(context_summary, context_lengths)
        self_attended_context = self.self_attentive_encoder_context(condensed_context, padding=context_padding)
        final_context, (context_rnn_h, context_rnn_c) = self.bilstm_context(self_attended_context[-1], context_lengths)
        context_rnn_state = [self.reshape_rnn_state(x) for x in (context_rnn_h, context_rnn_c)]

#        question_summary = self.dropout(torch.cat([coattended_question, question_encoded, question_embedded], -1))
        question_summary = torch.cat([coattended_question, question_encoded, question_embedded], -1)
        condensed_question, _ = self.question_bilstm_after_coattention(question_summary, question_lengths)
        self_attended_question = self.self_attentive_encoder_question(condensed_question, padding=question_padding)
        final_question, (question_rnn_h, question_rnn_c) = self.bilstm_question(self_attended_question[-1], question_lengths)
        question_rnn_state = [self.reshape_rnn_state(x) for x in (question_rnn_h, question_rnn_c)]

        #context_indices = context_limited if context_limited is not None else context
        #question_indices = question_limited if question_limited is not None else question
        #answer_indices = answer_limited if answer_limited is not None else answer
        
        #print('answer_indices')
        context_padding = context_indices.data == pad_idx
        question_padding = question_indices.data == pad_idx

        self.dual_ptr_rnn_decoder.applyMasks(context_padding, question_padding)

        if self.training:
            #answer_padding = (answer_indices.data == pad_idx)[:, :-1]
            #print(answer_indices.data , pad_idx)
            #print('0000000000000000000000000000000000000000000')
            #print('answer_padding',answer_padding)
            answer_padding = (answer_indices.data == pad_idx)[:, :-1]
            answer_embedded = self.decoder_embeddings(answer)
            #print(answer,answer_indices)
            #answer_embedded1=      
            #answer_padding1=torch.cat((answer_padding, answer_padding,answer_padding,answer_padding), 1)
            #print(answer[:, :-1].contiguous())
            self_attended_decoded = self.self_attentive_decoder(answer_embedded[:,:-1 ].contiguous(), self_attended_context, context_padding=context_padding, answer_padding=answer_padding, positional_encodings=True)
            
            decoder_outputs = self.dual_ptr_rnn_decoder(self_attended_decoded, 
                final_context, final_question, hidden=context_rnn_state)
            rnn_output, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switch, context_question_switch, rnn_state = decoder_outputs

            #p(Wt)
            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, context_question_switch,       
                context_attention, question_attention, 
                context_indices, question_indices, 
                oov_to_limited_idx)

            #print('probs',probs)

            #print('train_preds')
            pred_probs, preds = probs.max(-1)
            #print('===================pred_probs, preds =============',pred_probs, preds )

            
            #out=Variable(preds.data.cpu().apply_(self.map_to_full), volatile=True)
            #out=Variable(preds.data.cpu(), volatile=True)
            #print( '===================outs =============',out.size(),out)

            #print('answer_indices[:, 1:].contiguous()',answer_indices[:, 1:].contiguous())
            #answer_indices=torch.cat((answer_indices[:, 1:].contiguous(), answer_indices[:, 1:].contiguous(),answer_indices[:, 1:].contiguous(),answer_indices[:, 1:].contiguous()), 1)

            probs, targets = mask(answer_indices[:, :-1].contiguous(), probs.contiguous(), pad_idx=pad_idx)

            #print('===================probs, targets =============',probs )
            #print('=========================probs.log()=========================',probs.log())
            loss = F.nll_loss(probs.log(), targets)

            return loss, None
        else:
            return None, self.greedy(self_attended_context, final_context, final_question, 
                context_indices, question_indices,
                oov_to_limited_idx,rnn_state=context_rnn_state)
 
    def reshape_rnn_state(self, h):
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2).contiguous()

    def probs(self, generator, outputs, vocab_pointer_switches, context_question_switches, 
        context_attention, question_attention, 
        context_indices, question_indices, 
        oov_to_limited_idx):

        size = list(outputs.size())

        size[-1] = self.generative_vocab_size
        scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim()-1)
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = self.generative_vocab_size + len(oov_to_limited_idx)
        if self.generative_vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = Variable(scaled_p_vocab.data.new(*size).fill_(EPSILON))
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim()-1)

        p_context_ptr = Variable(scaled_p_vocab.data.new(*scaled_p_vocab.size()).fill_(EPSILON))
        p_context_ptr.scatter_add_(p_context_ptr.dim()-1, context_indices.unsqueeze(1).expand_as(context_attention), context_attention)
        scaled_p_context_ptr = (context_question_switches * (1 - vocab_pointer_switches)).expand_as(p_context_ptr) * p_context_ptr

        p_question_ptr = Variable(scaled_p_vocab.data.new(*scaled_p_vocab.size()).fill_(EPSILON))
        p_question_ptr.scatter_add_(p_question_ptr.dim()-1, question_indices.unsqueeze(1).expand_as(question_attention), question_attention)
        scaled_p_question_ptr = ((1 - context_question_switches) * (1 - vocab_pointer_switches)).expand_as(p_question_ptr) * p_question_ptr

        probs = scaled_p_vocab + scaled_p_context_ptr + scaled_p_question_ptr
        return probs


    def greedy(self, self_attended_context, context, question, context_indices, question_indices, oov_to_limited_idx, answer_indices=None,rnn_state=None):
        '''
        功能：测试
             由于测试部分没有answer,paper里把这部分吧answer都设为pad_idx=1
        '''
        B, TC, C = context.size()
        T = 3                       
        pad_idx=1
        outs = Variable(context.data.new(B, T).long().fill_(pad_idx), volatile=True)   
        #print(outs)                                             

        hiddens = [Variable(self_attended_context[0].data.new(B, T, C).zero_(), volatile=True)
                   for l in range(len(self.self_attentive_decoder.layers) + 1)]
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])

        eos_yet = context.data.new(B).byte().zero_()
        rnn_output, context_alignment, question_alignment = None, None, None
        for t in range(T):
            if t == 0:
                embedding = self.decoder_embeddings(Variable(
                    self_attended_context[-1].data.new(B).long().fill_(
                    2), volatile=True).unsqueeze(1), [1]*B)                                                                      #  t=0时，embedding都为一个数
                #embedding = self.decoder_embeddings(outs[:, t - 1].unsqueeze(1), [1]*B)
            else:
                embedding = self.decoder_embeddings(outs[:, t - 1].unsqueeze(1), [1]*B)

            #print('self.self_attentive_decoder.d_model',self.self_attentive_decoder.d_model)                                   #self.self_attentive_decoder.d_model=200
            hiddens[0][:, t] = hiddens[0][:, t] + (math.sqrt(self.self_attentive_decoder.d_model) * embedding).squeeze(1)       #设置Intermediate Decoder State  里 Aproj(-1)初始值
            #hiddens[0][:, t] =  (math.sqrt(self.self_attentive_decoder.d_model) * embedding).squeeze(1) 
            for l in range(len(self.self_attentive_decoder.layers)):
                hiddens[l + 1][:, t] = self.self_attentive_decoder.layers[l].feedforward(
                    self.self_attentive_decoder.layers[l].attention(
                    self.self_attentive_decoder.layers[l].selfattn(hiddens[l][:, t], hiddens[l][:, :t + 1], hiddens[l][:, :t + 1])
                  , self_attended_context[l], self_attended_context[l]))                                                       #输出Aself(t)

            decoder_outputs = self.dual_ptr_rnn_decoder(hiddens[-1][:, t].unsqueeze(1),
                context, question, 
                context_alignment=context_alignment, question_alignment=question_alignment,
                hidden=rnn_state, output=rnn_output)
            rnn_output, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switch, context_question_switch, rnn_state = decoder_outputs
            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, context_question_switch, 
                context_attention, question_attention, 
                context_indices, question_indices, 
                oov_to_limited_idx)    
            #print(answer_indices)
            if answer_indices != None:
                print('afadsadsadasdsa')
                probs=torch.cat((probs, probs), 0)                                                                                       #输出p(Wt)
                print(probs.view(-1,2,14696).contiguous())
                probs, targets = mask(answer_indices[:, 1:].contiguous(), probs.view(-1,2,14696).contiguous(), pad_idx=pad_idx)
                loss = F.nll_loss(probs.log(), targets)
                print('loss',loss)
            #print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',probs)
            pred_probs, preds = probs.max(-1)                                                                                

            #print('=================pred_probs, preds=====================',probs[:10])
            #print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',preds)
            eos_yet = eos_yet | (preds.data == 3)
            #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            #print(probs.view(500,-1).contiguous().log())

            #probs, targets = mask(answer_indices[:, 1:].contiguous(), probs.contiguous(), pad_idx=1)
            outs[:, t] = Variable(preds.data.cpu().apply_(self.map_to_full), volatile=True)
            #outs[:, t]=pred_probs
            #print('test_preds')
            #print( torch.unsqueeze(outs[:12,t], 0))
            if eos_yet.all():
                break
        return outs


class CoattentiveLayer(nn.Module):

    def __init__(self, d, dropout=0.2):
        super().__init__()
        self.proj = Feedforward(d, d, dropout=0.0)
        self.embed_sentinel = nn.Embedding(2, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, question, context_padding, question_padding): 
        context_padding = torch.cat([context.data.new(context.size(0)).long().fill_(0).unsqueeze(1).long()==1, context_padding], 1)
        question_padding = torch.cat([question.data.new(question.size(0)).long().fill_(0).unsqueeze(1)==1, question_padding], 1)

        context_sentinel = self.embed_sentinel(Variable(context.data.new(context.size(0)).long().fill_(0)))
        context = torch.cat([context_sentinel.unsqueeze(1), self.dropout(context)], 1) # batch_size x (context_length + 1) x features

        question_sentinel = self.embed_sentinel(Variable(question.data.new(question.size(0)).long().fill_(1)))
        question = torch.cat([question_sentinel.unsqueeze(1), question], 1) # batch_size x (question_length + 1) x features
        question = F.tanh(self.proj(question)) # batch_size x (question_length + 1) x features

        affinity = context.bmm(question.transpose(1,2)) # batch_size x (context_length + 1) x (question_length + 1)
        attn_over_context = self.normalize(affinity, context_padding) # batch_size x (context_length + 1) x 1
        attn_over_question = self.normalize(affinity.transpose(1,2), question_padding) # batch_size x (question_length + 1) x 1
        sum_of_context = self.attn(attn_over_context, context) # batch_size x (question_length + 1) x features
        sum_of_question = self.attn(attn_over_question, question) # batch_size x (context_length + 1) x features
        coattn_context = self.attn(attn_over_question, sum_of_context) # batch_size x (context_length + 1) x features
        coattn_question = self.attn(attn_over_context, sum_of_question) # batch_size x (question_length + 1) x features
        return torch.cat([coattn_context, sum_of_question], 2)[:, 1:], torch.cat([coattn_question, sum_of_context], 2)[:, 1:]

    @staticmethod
    def attn(weights, candidates):
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)).sum(1).squeeze(1)

    @staticmethod
    def normalize(original, padding):
        raw_scores = original.clone()
        raw_scores.data.masked_fill_(padding.unsqueeze(-1).expand_as(raw_scores), -INF)
        return F.softmax(raw_scores, dim=1)
        

class DualPtrRNNDecoder(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.0, num_layers=1):
        super().__init__()
        self.d_hid = d_hid
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.input_feed = True
        if self.input_feed:
            d_in += 1 * d_hid

        self.rnn = LSTMDecoder(self.num_layers, d_in, d_hid, dropout)
        self.context_attn = LSTMDecoderAttention(d_hid, dot=True)
        self.question_attn = LSTMDecoderAttention(d_hid, dot=True)

        self.vocab_pointer_switch = nn.Sequential(Feedforward(2 * self.d_hid + d_in, 1), nn.Sigmoid())
        self.context_question_switch = nn.Sequential(Feedforward(2 * self.d_hid + d_in, 1), nn.Sigmoid())

    def forward(self, input, context, question, output=None, hidden=None, context_alignment=None, question_alignment=None):
        context_output = output.squeeze(1) if output is not None else self.make_init_output(context)
        context_alignment = context_alignment if context_alignment is not None else self.make_init_output(context)
        question_alignment = question_alignment if question_alignment is not None else self.make_init_output(question)

        context_outputs, vocab_pointer_switches, context_question_switches, context_attentions, question_attentions, context_alignments, question_alignments = [], [], [], [], [], [], []
        for emb_t in input.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            context_output = self.dropout(context_output)
            if self.input_feed:
                emb_t = torch.cat([emb_t, context_output], 1)
            dec_state, hidden = self.rnn(emb_t, hidden)
            context_output, context_attention, context_alignment = self.context_attn(dec_state, context)
            question_output, question_attention, question_alignment = self.question_attn(dec_state, question)
            vocab_pointer_switch = self.vocab_pointer_switch(torch.cat([dec_state, context_output, emb_t], -1))
            context_question_switch = self.context_question_switch(torch.cat([dec_state, question_output, emb_t], -1))
            context_output = self.dropout(context_output)
            context_outputs.append(context_output)
            vocab_pointer_switches.append(vocab_pointer_switch)
            context_question_switches.append(context_question_switch)
            context_attentions.append(context_attention)
            context_alignments.append(context_alignment)
            question_attentions.append(question_attention)
            question_alignments.append(question_alignment)

        context_outputs, vocab_pointer_switches, context_question_switches, context_attention, question_attention = [self.package_outputs(x) for x in [context_outputs, vocab_pointer_switches, context_question_switches, context_attentions, question_attentions]]
        return context_outputs, context_attention, question_attention, context_alignment, question_alignment, vocab_pointer_switches, context_question_switches, hidden


    def applyMasks(self, context_mask, question_mask):
        self.context_attn.applyMasks(context_mask)
        self.question_attn.applyMasks(question_mask)

    def make_init_output(self, context):
        batch_size = context.size(0)
        h_size = (batch_size, self.d_hid)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def package_outputs(self, outputs):
        outputs = torch.stack(outputs, dim=1)
        return outputs
