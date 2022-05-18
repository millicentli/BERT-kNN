import numpy as np
import torch

from bert_knn.modules.base_connector import *
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5Config


class T5(Base_Connector):
    def __init__(self, args):
        super().__init__()

        t5_model_name = args.t5_model_name

        # Load the tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

        # Get the vocab
        # TODO: sanity check this
        # self.vocab = self.tokenizer.get_vocab()
        # Maps vocab to index
        self.inverse_vocab = self.tokenizer.get_vocab()
        # Maps index to vocab
        self.vocab = {self.inverse_vocab[word]: word for word in self.inverse_vocab}

        # Get the config before loading the model...
        config = T5Config.from_pretrained(t5_model_name, output_hidden_states=True)

        # Load the model to get output hidden states
        self.t5_model = T5Model.from_pretrained(t5_model_name, config=config)
        self.t5_model.eval()

        # Load the other model for generation
        self.t5_generate = T5ForConditionalGeneration.from_pretrained(t5_model_name)

        # Get the ids
        self.pad_id = self.inverse_vocab[T5_PAD]
        self.unk_id = self.inverse_vocab[T5_UNK]
        self.eos_id = self.inverse_vocab[T5_EOS]
    
    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        return indexed_string

    def _cuda(self):
        self.t5_model.cuda()
        self.t5_generate.cuda()

    def __get_input_tensors(self, sentences):
        if len(sentences) > 2:
            print(sentences)
            raise ValueError("More than two sentences in input")

        # Tokenize and append input ids, decoder ids
        # first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        # first_tokenized_sentence.append(T5_SEP)
        first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        # first_tokenized_decoder_ids = self.tokenizer.tokenize(sentences[0].insert(0, T5_PAD))
        first_tokenized_sentence_ids = self.tokenizer(sentences[0])

        if len(sentences) > 1:
            tokenized_text = ""
            assert len(sentences) != 1, "There are two sentences in the input!"
        else:
            tokenized_text = first_tokenized_sentence

        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == MASK_T5:
                masked_indices.append(i)
                break
        
        # Asserts to check some stuff
        assert len(first_tokenized_sentence_ids.input_ids) < 512, "Tokens are greater than len(512)!"
        assert masked_indices[0] == first_tokenized_sentence_ids.input_ids.index(self.inverse_vocab[MASK_T5]), "Tokenization for masking is not the same!"

        # Convert to PyTorch tensors
        tokens_tensor = torch.tensor([first_tokenized_sentence_ids["input_ids"]])
        attention_mask = torch.tensor([first_tokenized_sentence_ids["attention_mask"]])

        return tokens_tensor, attention_mask, masked_indices, tokenized_text

    def __get_input_tensors_batch(self, sentences_list):
        tokens_tensors_list = []
        masked_indices_list = []
        attention_mask_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, attention_mask, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor)
            masked_indices_list.append(masked_indices)
            attention_mask_list.append(attention_mask)
            tokenized_text_list.append(tokenized_text)

            if tokens_tensor.shape[1] > max_tokens:
                max_tokens = tokens_tensor.shape[1]
        
        final_tokens_tensor = None
        final_attention_mask = None
        for tokens_tensor, attention_mask in zip(tokens_tensors_list, attention_mask_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_length = max_tokens - dim_tensor
            attention_tensor = torch.full([1, dim_tensor], 1, dtype=torch.long)
            if pad_length > 0:
                pad = torch.full([1, pad_length], self.pad_id, dtype=torch.long)
                attention_pad = torch.full([1, pad_length], 0, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)
            # TODO: remove this if statement. Seems kind of useless?
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
        # print("Here's final_tokens_tensor:", final_tokens_tensor)
        # print("Here's final_attention_mask:", final_attention_mask)
        # print("Here's masked_indices_list:", masked_indices_list)
        # print("Here's tokenized_text_list:", tokenized_text_list)
        return final_tokens_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    def get_hidden_state(self, sentences_list, try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, attention_mask_tensor, masked_indices_list, _ = \
            self.__get_input_tensors_batch(sentences_list)
    
        with torch.no_grad():
            ids = tokens_tensor.to(self._model_device)
            output = self.t5_model(
                input_ids=ids,
                decoder_input_ids=ids,
                attention_mask=attention_mask_tensor.to(self._model_device),
                return_dict=True
            )

        # Similar to the paper, try the layer right before the last one
        hidden = output.encoder_hidden_states[-2]
        # hidden = hidden[-2]
        hidden = hidden[np.arange(hidden.shape[0]), np.array(masked_indices_list).flatten()]
        
        hidden = hidden.cpu()
        return hidden

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()
        
        tokens_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)
        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))
        
        with torch.no_grad():
            ids = tokens_tensor.to(self._model_device)
            logits = self.t5_generate(
                input_ids=ids,
                decoder_input_ids=ids,
                attention_mask=attention_mask_tensor.to(self._model_device),
                return_dict=True
            ).logits
        print("Here's logits shape:", logits.shape)

        masked_output = logits[np.arange(logits.shape[0]), np.array(masked_indices_list).flatten()]
        masked_output = torch.softmax(masked_output, dim=-1).cpu()

        return masked_output, masked_indices_list
