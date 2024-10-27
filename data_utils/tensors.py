import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        # if it's a text-only dataset, 'inp' should still be of max_len to enable correct masking in 'collate'
        'inp': torch.tensor(b[4].T).float().unsqueeze(1) if b[4].ndim > 1 else torch.zeros(b[5]), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)

# an adapter to MDM's collate func for the Motion Transfer Benchmark (MTB)
def t2m_transfer_collate(batch):
    # batch is built from (leader, follower) pairs
    adapted_batch = [{
        # 'inp': torch.tensor(b[4][0].T).float().unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
        'leader_inp': torch.tensor(b[4][0].T).float().unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
        'follower_inp': torch.tensor(b[4][1].T).float().unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
        'leader_text': b[2][0],
        'follower_text': b[2][1],
        'leader_tokens': b[6][0],
        'follower_tokens': b[6][1],
        'leader_lengths': b[5][0],
        'follower_lengths': b[5][1],
        'leader_idx': b[7][0],
        'follower_idx': b[7][1],
    } for b in batch]
    return transfer_collate(adapted_batch)

def data_mask_process(notnone_batches, prefix):
    databatch = [b[f'{prefix}_inp'] for b in notnone_batches]
    lenbatch = [b[f'{prefix}_lengths'] for b in notnone_batches]
    databatchTensor = collate_tensors(databatch)
    max_len = databatchTensor.shape[-1]
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, max_len).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
    assert lenbatchTensor.max() <= databatchTensor.shape[-1]
    idx = [b[f'{prefix}_idx'] for b in notnone_batches]
    return {f'{prefix}_mask': maskbatchTensor, f'{prefix}_lengths': lenbatchTensor, f'{prefix}_motion': databatchTensor, f'{prefix}_idx': idx}


def transfer_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    cond = {'y': {}}
    cond['y'].update(data_mask_process(notnone_batches, 'leader'))
    cond['y'].update(data_mask_process(notnone_batches, 'follower'))

    for k in notnone_batches[0].keys():
        if 'text' in k or 'tokens' in k:
            textbatch = [b[k] for b in notnone_batches]
            cond['y'].update({k: textbatch})

    return None, cond
