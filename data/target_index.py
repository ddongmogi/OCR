from jamo import h2j, j2hcj
from data.unicode import join_jamos

def phoneme_index(paragraph_form=True):
    JAMO_LEADS = [chr(_) for _ in range(0x1100, 0x115F)]
    JAMO_LEADS_MODERN = [chr(_) for _ in range(0x1100, 0x1113)]
    JAMO_VOWELS = [chr(_) for _ in range(0x1161, 0x11A8)]
    JAMO_VOWELS_MODERN = [chr(_) for _ in range(0x1161, 0x1176)]
    JAMO_TAILS = [chr(_) for _ in range(0x11A8, 0x1200)]
    JAMO_TAILS_MODERN = [chr(_) for _ in range(0x11A8, 0x11C3)]

    jamo_list = JAMO_LEADS_MODERN+JAMO_VOWELS_MODERN+JAMO_TAILS_MODERN

    jamo2ind = {}
    ind2jamo = {}
    
    for i,jm in enumerate(jamo_list):
        ind2jamo[i] = jm
        jamo2ind[jm] = i
        
    jamo2ind['\t']= 67
    ind2jamo[67] = '\t'
    
    if paragraph_form:
        jamo2ind['\n'] = 68
        ind2jamo[68] = '\n'
    
    return jamo2ind, ind2jamo


def character_index(path,paragraph_form=True):
    import json

    with open(path,'r') as f:
        js = json.load(f)[:-1]

    js.append('\t')
    
    if paragraph_form:
        js.append('\n')

    char2ind={}
    ind2char={}

    for i,jm in enumerate(js):
        char2ind[jm]=i
        ind2char[i]=jm
    
    return char2ind, ind2char