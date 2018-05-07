import numpy as np

mr_types = ['name', 'eatType', 'food', 'priceRange', 'customer rating', 'area', 'familyFriendly', 'near']

# Constructs a mapping from a meaning representation type/value to an integer which
#  can be used to encode the MR into a feature vector
def construct_type_mapping(meaning_representations):

    # Get the possible 'type' configurations from the raw meaning representations
    d = {}
    for s in meaning_representations:
        comps = s.split(',')
        for c in comps:
            for t in mr_types:
                c = c.strip()
                if c.startswith(t):
                    if t not in d:
                        d[t] = set()
                    
                    val = c[len(t)+1:].replace(']', '')
                    d[t].add(val)

    # Creates a mapping that converts the mr type to an Id for the feature vector
    type2id = {'name':0, 'near':1}
    i = 2
    for k, v in d.items():
        if k not in ['name', 'near']:
            for a in v:
                type2id[(k,a)] = i
                i += 1

    # Also add a 'not specified' component for all types that can be not specified (which are all but 'name')
    not_specified = ['eatType', 'food', 'priceRange', 'customer rating', 'area', 'familyFriendly', 'near']
    for a in not_specified:
        type2id[(a, 'not specified')] = len(type2id)

    return type2id

# Convert the string meaning representation to a list of tuples (or dictionary for that matter)
def structure_mr_string(mr_str):
    mr = []
    
    comps = mr_str.split(',')
    for c in comps:
        for t in mr_types:
            c = c.strip()
            if c.startswith(t):
                val = c[len(t)+1:].replace(']', '')
                mr.append((t, val))
    return mr

# Convert a given structered meaning representation to a feature vector
def to_feature_vector(structured_mrs, type2id):
    vec = np.zeros(len(type2id))
    
    specified = set()
    for k,v in structured_mrs:
        specified.add(k)
        if k in ['name', 'near']:
            vec[type2id[k]] = 1
        else:
            vec[type2id[(k,v)]] = 1
    
    # Add the non specified keys as well
    for not_specified in set(mr_types) - specified:
        vec[type2id[(k, 'not specified')]] = 1
    
    return vec

def construct_vocab(refs, structered_mrs):

    # Replace the name and near parts in the sentence with a placeholder
    proc_sents = []
    for i_s in range(len(refs)):
        s = refs[i_s]
        mr = structered_mrs[i_s]
        for k,v in mr:
            if k == 'name':
                s = s.replace(v, ' <name> ')
            elif k == 'near':
                s = s.replace(v, ' <near> ')
        proc_sents.append(s.lower())


    vocab = {'<name>', '<near>', '<bos>', '<eos>'}
    for s in proc_sents:
        
        # for every c=character in s=sentence
        for c in s:
            vocab.update(c)

    return list(vocab)

# Converts a reference sentence by mapping all the characters to its corresponding id
def convert_ref(ref, char2id):
    sent_ids = [char2id['<bos>']]
    
    comps = ref.split(' ')
    for i in range(len(comps)):
        word = comps[i]
        
        if word == '<name>':
            sent_ids.append(char2id['<name>'])
        elif word == '<near>':
            sent_ids.append(char2id['<near>'])
        else:
            # For c=character in word
            for c in word:
                sent_ids.append(char2id[c.lower()])
                
            # Don't add a whitespace after the last word
            if i < len(comps) - 1:
                sent_ids.append(char2id[' '])
            
    sent_ids.append(char2id['<eos>'])
    return sent_ids

# Returns a matrix for 
def one_hot_ref(r, max_seq_len, vocab):
    S = np.zeros((max_seq_len, len(vocab)))
    for j in range(len(r)):
        if j >= len(vocab):
            break
        
        vec = np.zeros(len(vocab))
        vec[r[j]] = 1
        S[j] = vec
    return S