from xml.etree.ElementTree import QName
import torch
import transformers
from transformers import BertTokenizer, BertForMaskedLM, BasicTokenizer, BertModel, BertConfig
import numpy as np
from bert_knn.modules.base_connector import *
import torch.nn.functional as F

class Bert(Base_Connector):

    def __init__(self, args):
        super().__init__()

        bert_model_name = args.bert_model_name
        # When using a cased model, make sure to pass do_lower_case=False directly to BaseTokenizer
        do_lower_case = False
        if 'uncased' in bert_model_name:
            do_lower_case=True

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # original vocab
        self.map_indices = None
        self.vocab = list(self.tokenizer.ids_to_tokens.values())
        self._init_inverse_vocab()

        # Load pre-trained model (weights)
        # ... to get prediction/generation

        config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        self.masked_bert_model = BertForMaskedLM.from_pretrained(bert_model_name, config=config)

        self.masked_bert_model.eval()

        # ... to get pooled output
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_model.eval()

        # ... to get hidden states
        self.bert_model_hidden = BertModel.from_pretrained(bert_model_name, config=config)
        self.bert_model_hidden.eval()

        self.pad_id = self.inverse_vocab[BERT_PAD]

        self.unk_index = self.inverse_vocab[BERT_UNK]

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)

        return indexed_string

    def __get_input_tensors_batch(self, sentences_list):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, segments_tensor, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            # assert(tokens_tensor.shape[1] == segments_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # print("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_length = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype=torch.long)
            if pad_length > 0:
                pad_1 = torch.full([1,pad_length], self.pad_id, dtype=torch.long)
                pad_2 = torch.full([1,pad_length], 0, dtype=torch.long)
                attention_pad = torch.full([1,pad_length], 0, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_segments_tensor = segments_tensor
                final_attention_mask = attention_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0)

        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    def __get_input_tensors(self, sentences):
        if len(sentences) > 2:
            print(sentences)
            raise ValueError("BERT accepts maximum two sentences in input for each data point")

        first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        # add [SEP] token at the end
        first_tokenized_sentence.append(BERT_SEP)
        first_segment_id.append(0)

        if len(sentences)>1 :
            second_tokenized_sentece = self.tokenizer.tokenize(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            # add [SEP] token at the end
            second_tokenized_sentece.append(BERT_SEP)
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        # add [CLS] token at the beginning
        tokenized_text.insert(0, BERT_CLS)
        segments_ids.insert(0,0)

        # print("Here's sentences:", sentences)
        # print("Here's the tokenized_text:", tokenized_text)
        # look for masked indices
        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == MASK:
                # TODO: fix this, it's adding more than one token for some reason???????
                masked_indices.append(i)
                break

        max_tokens = 512
        if len(tokenized_text) > max_tokens:
            shift = int(max_tokens/2)
            if masked_indices[0] > shift:
                start =  masked_indices[0]-shift
                end = masked_indices[0]+shift
                masked_indices[0] = shift
            else:
                start = 0
                end = max_tokens
            segments_ids = segments_ids[start:end]
            tokenized_text = tokenized_text[start:end]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text

    def __get_input_tensors_multi_token(self, sentence):

        assert len(sentence) < 2
        
        sents_tokenized = self.tokenizer(sentence, return_tensors="pt", padding=True)
        tokenized_text = self.tokenizer.tokenize(sentence[0])

        assert len(sents_tokenized["input_ids"][0]) == len(tokenized_text) + 2

        tokenized_text = sents_tokenized["input_ids"]
        segment_ids = sents_tokenized["token_type_ids"]
        attention_mask = sents_tokenized["attention_mask"]

        masked_indices = []
        for i in range(len(tokenized_text[0])):
            token = tokenized_text[0][i]
            if token == self.tokenizer.mask_token_id:
                masked_indices.append(i)

        max_tokens = 512
        if len(tokenized_text) > max_tokens:
            shift = int(max_tokens/2)
            if masked_indices[0] > shift:
                start =  masked_indices[0]-shift
                end = masked_indices[0]+shift
                masked_indices[0] = shift
            else:
                start = 0
                end = max_tokens
            segment_ids = segment_ids[start:end]
            tokenized_text = tokenized_text[start:end]

        # Convert inputs to PyTorch tensors
        tokens_tensor = tokenized_text
        segment_ids = segment_ids
        attention_mask = attention_mask
        # tokens_tensor = torch.tensor([indexed_tokens])
        # segments_tensors = torch.tensor([segment_ids])

        # return tokens_tensor, segment_tensors, masked_indices, tokenized_text
        return tokens_tensor, segment_ids, attention_mask, masked_indices, tokenized_text


    def __get_token_ids_from_tensor(self, indexed_string):
        token_ids = []
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)
            token_ids = np.asarray(indexed_string)
        else:
            token_ids = indexed_string
        return token_ids

    def _cuda(self):
        self.masked_bert_model.cuda()
        self.bert_model.cuda()
        self.bert_model_hidden.cuda()

    def get_batch_generation(self, sentences_list, logger= None,
                             try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))

        with torch.no_grad():
            logits = self.masked_bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )

            #log_probs = F.log_softmax(logits[0], dim=-1).cpu()
            all_output = logits[0]

        masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        masked_output = torch.softmax(masked_output, dim=-1).cpu()

        #log_probs = predictions[0, masked_indices_list]
        """token_ids_list = []
        for indexed_string in tokens_tensor.numpy():
            token_ids_list.append(self.__get_token_ids_from_tensor(indexed_string))"""

        return masked_output, masked_indices_list

    def get_generation_multi_token(self, sentences_list, logger=None, try_cuda=True, k=100):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()
             
        logits_list = []
        masked_indices_list = []
        hiddens_list = []
        max_dim = -1
        for sentence in sentences_list:
            tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices, tokenized_text = self.__get_input_tensors_multi_token(sentence)

            if logger is not None:
                logger.debug("\n{}\n".format(tokenized_text))

            masked_indices_list.append(masked_indices)
            tens_list = []
            hidden_list = []
            for idx, mask in enumerate(masked_indices):
                if idx == 0:
                    with torch.no_grad():
                        outputs = self.masked_bert_model(
                            input_ids=tokens_tensor.to(self._model_device),
                            token_type_ids=segments_tensor.to(self._model_device),
                            attention_mask=attention_mask_tensor.to(self._model_device),
                        )
                        predictions = outputs[0]
                    
                        _, sorted_idx = predictions[0].sort(dim=-1, descending=True)
                        sorted_idx_sliced = sorted_idx[mask, :]
                        best_cands = [sorted_idx_sliced[i].item() for i in range(k)]
                        # breakpoint()
                        hidden_sliced = outputs[-1][-2][:, mask, :]
                        hidden_list.append(hidden_sliced)

                        # best_pred_cands = [self.tokenizer.convert_ids_to_tokens(cand) for cand in best_cands]

                        # Create 100 new sentences with the best options
                        
                        for cand in best_cands:
                            new_tensor = tokens_tensor.clone().detach()
                            new_tensor[:, mask] = cand
                            tens_list.append(new_tensor)
                else:
                    # new_list = np.array(tens_list.copy()).to(self._model_device)
                    if not torch.is_tensor(tens_list):
                        inputs = torch.stack(tens_list).squeeze().to(self._model_device)
                        with torch.no_grad():
                            outputs = self.masked_bert_model(inputs)
                        tens_list.clear()
                    else:
                        with torch.no_grad():
                            outputs = self.masked_bert_model(inputs)
                    predictions = outputs[0]
                    
                    # breakpoint()
                    hidden_sliced = outputs[-1][-2][:, mask, :]
                    hidden_list.append(hidden_sliced)

                    for i in range(len(predictions)):
                        _, sorted_idx = predictions[i].sort(dim=-1, descending=True)
                        sorted_idx_sliced = sorted_idx[mask, :]
                        best_cand = sorted_idx_sliced[0].item()

                        
                        # new_tensor = inputs[i].clone().detach()
                        inputs[i, mask] = best_cand

                    tens_list = inputs

            logits = outputs[0].log_softmax(-1)
            scores = torch.gather(logits, 2, tens_list[:, :, None]).squeeze(-1)
            unique_scores = scores.sum(-1)

            _, sorted_idx = unique_scores.sort(descending=True)


            # Sanity checking
            # for idx, i in enumerate(sorted_idx):
            #     print(f"Top {idx}: {tens_list[i]}")
            #     print("Checking the decoding:", self.tokenizer.decode(tens_list[i]))

            # Just add the best one
            best_sent = outputs[0][sorted_idx[0], :, :]
            masked_output = best_sent[np.array(masked_indices).flatten()]
            masked_output = torch.softmax(masked_output, dim=-1).cpu()

            logits_list.append(masked_output)
            for idx in range(1, len(hidden_list)):
                hidden_list[idx] = torch.unsqueeze(hidden_list[idx][sorted_idx[0]], 0)

            # Do some averaging
            best_hidden = torch.squeeze(torch.stack(hidden_list, 0))
            # best_hidden = torch.stack(hidden_list)
            # best_hidden = torch.mean(best_hidden, dim=0)
            # best_hidden = torch.unsqueeze(best_hidden, 0)
            hiddens_list.append(best_hidden)

            max_dim = max(max_dim, len(best_hidden))

        hiddens_list = [torch.cat((l, torch.full((max_dim - len(l), 768), -100).cuda()), 0) if len(l) < max_dim else l for l in hiddens_list]
        hiddens_list = torch.squeeze(torch.stack(hiddens_list), 1).cpu()
        return logits_list, masked_indices_list, hiddens_list

    def get_generation_multi_token_averaged(self, sentences_list, logger=None, try_cuda=True, k=100):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()
             
        logits_list = []
        masked_indices_list = []
        hiddens_list = []
        max_dim = -1
        for sentence in sentences_list:
            tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices, tokenized_text = self.__get_input_tensors_multi_token(sentence)

            if logger is not None:
                logger.debug("\n{}\n".format(tokenized_text))

            masked_indices_list.append(masked_indices)
            tens_list = []
            hidden_list = []
            for idx, mask in enumerate(masked_indices):
                if idx == 0:
                    with torch.no_grad():
                        outputs = self.masked_bert_model(
                            input_ids=tokens_tensor.to(self._model_device),
                            token_type_ids=segments_tensor.to(self._model_device),
                            attention_mask=attention_mask_tensor.to(self._model_device),
                        )
                        predictions = outputs[0]
                    
                        _, sorted_idx = predictions[0].sort(dim=-1, descending=True)
                        sorted_idx_sliced = sorted_idx[mask, :]
                        best_cands = [sorted_idx_sliced[i].item() for i in range(k)]
                        # breakpoint()
                        hidden_sliced = outputs[-1][-2][:, mask, :]
                        hidden_list.append(hidden_sliced)

                        # best_pred_cands = [self.tokenizer.convert_ids_to_tokens(cand) for cand in best_cands]

                        # Create 100 new sentences with the best options
                        
                        for cand in best_cands:
                            new_tensor = tokens_tensor.clone().detach()
                            new_tensor[:, mask] = cand
                            tens_list.append(new_tensor)
                else:
                    # new_list = np.array(tens_list.copy()).to(self._model_device)
                    if not torch.is_tensor(tens_list):
                        inputs = torch.stack(tens_list).squeeze().to(self._model_device)
                        with torch.no_grad():
                            outputs = self.masked_bert_model(inputs)
                        tens_list.clear()
                    else:
                        with torch.no_grad():
                            outputs = self.masked_bert_model(inputs)
                    predictions = outputs[0]
                    
                    # breakpoint()
                    hidden_sliced = outputs[-1][-2][:, mask, :]
                    hidden_list.append(hidden_sliced)

                    for i in range(len(predictions)):
                        _, sorted_idx = predictions[i].sort(dim=-1, descending=True)
                        sorted_idx_sliced = sorted_idx[mask, :]
                        best_cand = sorted_idx_sliced[0].item()

                        
                        # new_tensor = inputs[i].clone().detach()
                        inputs[i, mask] = best_cand

                    tens_list = inputs

            logits = outputs[0].log_softmax(-1)
            scores = torch.gather(logits, 2, tens_list[:, :, None]).squeeze(-1)
            unique_scores = scores.sum(-1)

            _, sorted_idx = unique_scores.sort(descending=True)


            # Sanity checking
            # for idx, i in enumerate(sorted_idx):
            #     print(f"Top {idx}: {tens_list[i]}")
            #     print("Checking the decoding:", self.tokenizer.decode(tens_list[i]))

            # Just add the best one
            best_sent = outputs[0][sorted_idx[0], :, :]
            masked_output = best_sent[np.array(masked_indices).flatten()]
            masked_output = torch.softmax(masked_output, dim=-1).cpu()
            logits_list.append(masked_output)
            for idx in range(1, len(hidden_list)):
                hidden_list[idx] = torch.unsqueeze(hidden_list[idx][sorted_idx[0]], 0)

            # Do some averaging
            # best_hidden = torch.squeeze(torch.stack(hidden_list, 0))
            best_hidden = torch.squeeze(torch.stack(hidden_list), dim=0)
            best_hidden = torch.mean(best_hidden, dim=0)
            hiddens_list.append(best_hidden)

            max_dim = max(max_dim, len(best_hidden))

        hiddens_list = [torch.cat((l, torch.full((max_dim - len(l), 768), -100).cuda()), 0) if len(l) < max_dim else l for l in hiddens_list]
        hiddens_list = torch.squeeze(torch.stack(hiddens_list), 1).cpu()
        return logits_list, masked_indices_list, hiddens_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, _, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, pooled_output = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        pooled_output = pooled_output.cpu()
        # attention_mask_tensor = attention_mask_tensor.type(torch.bool)
        return _, pooled_output

    def get_contextual_embeddings_mean(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, _, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            all_embeddings, pooled_output = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        attention_mask_tensor = attention_mask_tensor.type(torch.bool)
        output = np.zeros((all_embeddings.shape[0], all_embeddings.shape[2]))
        for idx, (embeddings, attention_mask) in enumerate(zip(all_embeddings, attention_mask_tensor)):
            output[idx] = np.mean(np.array(embeddings[attention_mask].cpu()), axis=0)
        return output


    def get_contextual_embeddings_mask_token(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            all_output, _ = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        masked_output = masked_output.cpu()
        return masked_output

    def get_hidden_state(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, _, hidden = self.bert_model_hidden(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        hidden = hidden[-2]
        hidden = hidden[np.arange(hidden.shape[0]), np.array(masked_indices_list).flatten()]
        #masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        hidden = hidden.cpu()
        return hidden

    def get_hidden_state_3(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, _, hidden = self.bert_model_hidden(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        hidden = hidden[-3]
        hidden = hidden[np.arange(hidden.shape[0]), np.array(masked_indices_list).flatten()]
        #masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        hidden = hidden.cpu()
        return hidden

    def get_hidden_state_4(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, _, hidden = self.bert_model_hidden(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        hidden = hidden[-4]
        hidden = hidden[np.arange(hidden.shape[0]), np.array(masked_indices_list).flatten()]
        #masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        hidden = hidden.cpu()
        return hidden

    def get_NN(self, sentences_list):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, _, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, pooled_output = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        pooled_output = pooled_output.cpu()
        return pooled_output
