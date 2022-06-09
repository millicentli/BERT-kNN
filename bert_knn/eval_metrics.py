import torch
import numpy as np
import faiss
import json


def normalize(distances, n=8):
    if sum(distances) != 0.0:
        distances = (1/np.power(distances, n)/sum(1/np.power(distances, n)))
    return distances


def normalize_exp(distances, n=6):
    if sum(distances) != 0.0:
        distances = (np.exp(-distances/n)/sum(np.exp(-distances/n)))
    return distances


def dump_jsonl(data, output_path, append=True):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))


def ivecs_read(fname, count=-1, offset=0):
    a = np.fromfile(fname, dtype='int32', count=count, offset=offset)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname, count=-1, offset=0):
    return ivecs_read(fname, count=count, offset=offset).view('float32')


def get_bert_preds(predictions, topk=10):
    bert_probs = []
    bert_vocab = []

    for i in range(len(predictions)):
        probs_bert, vocab_idcs_bert = torch.topk(input=predictions[i], k=predictions.shape[1], dim=0)
        bert_probs.append(probs_bert)
        bert_vocab.append(vocab_idcs_bert)

    return bert_probs, bert_vocab

def interpolate(distances, labels, predictions, topk=10):
    # Get the resulting best bert
    probs_bert, vocab_idcs_bert = get_bert_preds(predictions)

    occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)

    # normalizes NN probs
    normalized_distances = normalize_exp(distances[0])
    normalized_distances = [[normalized_distances]]
    # Take the probability of the first by itself
    # Then take the probability of the next (use the same)
    unique_predictions = np.unique(labels)
    # Get the max total (so we know how much to extend the probs_vocab_nn)
    max_size = max(i[0] + i[1] * 2 for i in unique_predictions if len(i) > 1)
    # max_size = 0
    # for i in unique_predictions:
    #     print("i:", i)
    #     max_size = max(max_size, sum(i))
    
    d = predictions.shape[1] + max_size + 1
    probs_vocab_nn = torch.zeros(d)
    probs_vocab_bert = torch.zeros(d)
    breakpoint()
    mappings = {}
    # Calculate nn probs, bert probs here
    for p in unique_predictions:
        print("Here's p:", p)
        idcs_unique = list(occurrences(p, labels))
        if len(p) == 1:
            probs_vocab_nn[p] = sum(normalized_distances[0][0][idcs_unique])
        else:
            probs_vocab_nn[predictions.shape[1] + p[0] + p[1] * 2] = sum(normalized_distances[0][0][idcs_unique])
            # TODO: this is really dummy, it just takes the probability of the first token as the combined prob
            idx = (vocab_idcs_bert[0] == p[0]).nonzero(as_tuple=True)[0].item()
            # idx2 = (vocab_idcs_bert[1] == p[1]).nonzero(as_tuple=True)[0].item() 
            probs_vocab_bert[predictions.shape[1] + p[0] + p[1] * 2] = vocab_idcs_bert[0][idx]
            mappings[predictions.shape[1] + p[0] + p[1] * 2] = p
        # probs = 0.0
        # indices = []
        # for item in p:
        #     idx = (bert_vocab == item).nonzero(as_tuple=True)[0].item()
        #     indices.append(idx)
        #     probs += bert_probs[idx]
        # probs_vocab_bert[d + sum(p)] = 
    # Get the resulting best nn
    probs_nn, vocab_idcs_nn = torch.topk(input=probs_vocab_nn, k=topk, dim=0)

    # Get the combined preds
    weighted = 0.3
    # for bert_prob in probs_bert:
    # TODO: get the probability of specific two words [w_1, w_2] in the knn - that's the knn vocab
    probs_combined = weighted*probs_vocab_nn + (1-weighted)*probs_vocab_bert

    probs_combined, vocab_idcs_combined = torch.topk(input=probs_combined, k=topk, dim=0)

    # Combine the bert predictions for each label, create a new prob
    # For each prob, take both, get softmax and average over
    return vocab_idcs_combined, probs_combined, vocab_idcs_bert, probs_bert, vocab_idcs_nn, probs_nn, mappings


def get_ranking(predictions, log_probs, sample, vocab, ranker, labels_dict_id, labels_dict, label_index=None,
                index_list=None):
    P_AT_1 = 0.
    P_AT_1_nn = 0.
    P_AT_1_bert = 0.

    vocab_r = list(vocab.keys())

    labels = []
    sentences = []
    label_tokens = []

    all_bert_preds = []
    all_combined_preds = []
    all_nn_preds = []
    
    all_probs_bert = []
    all_probs_combined = []
    all_probs_nn = []

    experiment_result = {}
    return_msg = ""

    # path_vectors = "/private/home/millicentli/BERT-kNN/DrQA/data/vectors/vectors_dump_"
    path_vectors = "/private/home/millicentli/BERT-kNN/DrQA/data/test_multimasking_stuff/vectors/vectors_dump_"
    breakpoint()
    N = 128
    num_ids = 3
    d = 768
    if "sub_label" in sample:
        if sample["sub_label"] == "squad" or sample["sub_label"] == "templama":
            query = sample["masked_sentences"][0]
            query = query.replace("[MASK]", "")
            query = query.replace(".", "").strip()
        else:
            query = sample["sub_label"]
    elif "sub" in sample:
        query = sample["sub"]

    doc_names, doc_scores = ranker.closest_docs(query, num_ids)

    filtered = [(name, score) for (name, score) in zip(doc_names, doc_scores)]

    if query.lower() in labels_dict_id and query.lower() not in doc_names:
        filtered = [(query.lower(), 1.0)]
    all_idcs = []
    doc_weights = []
    index = faiss.IndexFlatL2(d)

    for name, score in filtered:
        if name.lower() in labels_dict_id:
            idcs = labels_dict_id[name.lower()]
            count = ((idcs[3]+1)-idcs[2])+((idcs[3]+1)-idcs[2])*d
            offset = (d+1)*idcs[2]
            xt = fvecs_read(path_vectors + str(idcs[0]) + ".fvecs", count=count, offset=4*offset)
            xt = np.array(xt)

            index.add(xt)

            label_idcs = [(idcs[0], c) for c in range(idcs[2], len(xt)+idcs[2])]
            all_idcs.extend(label_idcs)
            scores = [score]*len(xt)
            doc_weights.extend(scores)

    predictions = predictions.reshape(-1, 768)
    distances, top_k = index.search(np.array(predictions), N)
    idx_cut = len(top_k[0])
    for idx, (k, d) in enumerate(zip(top_k[0], distances[0])):
        if k == -1:
            idx_cut = idx
            break
        else:

            label_idx = all_idcs[k]
            instance_id = "{:02}_{:08}".format(label_idx[0], label_idx[1])
            label_token = labels_dict.get_labels(instance_id).strip()
            label_list = []
            for label in label_token.split(" "):
                label_vocab_idx = vocab[label]
                label_vocab_int = int(label_vocab_idx)
                label_list.append(label_vocab_int)
            labels.append(label_list)
            # labels.append(int(label_vocab_idx))
            sentences.append(label_idx)
            label_tokens.append(label_token)

    distances = [distances[0][0:idx_cut]]
    vocab_idcs_combined, probs_combined, vocab_idcs_bert, probs_bert, vocab_idcs_nn, probs_nn, mappings = \
        interpolate(distances, labels, log_probs)

    if label_index is not None:

        # check if the labe_index should be converted to the vocab subset
        if index_list is not None:
            label_index = index_list.index(label_index)
        if len(labels) > 0:
            if label_index == vocab_idcs_nn[0]:
                P_AT_1_nn = 1.
            if label_index == vocab_idcs_combined[0]:
                P_AT_1 = 1.
            if label_index in vocab_idcs_bert[0]:
                P_AT_1_bert = 1.

    out = []
    for bert in vocab_idcs_bert:
        l = []
        for idx, num in enumerate(bert.tolist()):
            if idx == 10:
                break
            l.append(vocab_r[num])
        out.append(l)

    predictions_combined = []
    for idx in vocab_idcs_combined.tolist():
        if idx > log_probs.shape[1] - 1:
            result = mappings[idx]
            word = " ".join([vocab_r[r] for r in result])
            predictions_combined.append(word)
        else:
            predictions_combined.append(vocab_r[idx])

    predictions_nn = []
    for idx in vocab_idcs_nn.tolist():
        if idx > log_probs.shape[1] - 1:
            result = mappings[idx]
            word = " ".join([vocab_r[r] for r in result])
            predictions_nn.append(word)
        else:
            predictions_nn.append(vocab_r[idx])

    predictions_bert = []
    probabilities_bert = []
    for l in zip(vocab_idcs_bert):
        for preds in l:
            out = preds[:10].tolist()
            for idx, pred in enumerate(out):  
                if len(predictions_bert) <= idx:
                    predictions_bert.append([vocab_r[pred]])
                else:
                    predictions_bert[idx].append(vocab_r[pred])

    for l in zip(probs_bert):
        for preds in l:
            out = preds[:10].tolist()
            for idx, pred in enumerate(out):  
                if len(probabilities_bert) <= idx:
                    probabilities_bert.append([pred])
                else:
                    probabilities_bert[idx].append(pred)
    experiment_result["topk_bert"] = predictions_bert
    experiment_result["topk_combined"] = predictions_combined
    experiment_result["topk_nn"] = predictions_nn
    experiment_result["probs_nn"] = probs_nn.tolist()
    experiment_result["probs_bert"] = probabilities_bert
    experiment_result["probs_combined"] = probs_combined.tolist()

    experiment_result["P_AT_1"] = P_AT_1
    experiment_result["P_AT_1_nn"] = P_AT_1_nn
    experiment_result["P_AT_1_bert"] = P_AT_1_bert
    experiment_result["documents"] = list(doc_names)

    experiment_result["document_scores"] = list(doc_scores)
    experiment_result["all_labels"] = label_tokens
    experiment_result["sample"] = sample["masked_sentences"]
    experiment_result["answer"] = sample["obj_label"]
    # experiment_result["generated_bert"] = ' '.join(topk_all[0])
    # experiment_result["generated_combined"] = ' '.join(topk_all[1])
    # experiment_result["generated_nn"] = ' '.join(topk_all[2])
    print("Here's experiment_result:", experiment_result)
    return experiment_result, return_msg
