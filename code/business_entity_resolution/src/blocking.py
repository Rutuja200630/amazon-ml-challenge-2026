import collections
import normalization as norm

COMMON_ADDR_WORDS = {
    # English
    'street', 'road', 'avenue', 'boulevard', 'drive', 'court', 'lane', 'place',
    'circle', 'way', 'trail', 'parkway', 'highway', 'suite', 'floor', 'apartment',
    'building', 'room', 'number', 'near', 'opposite', 'behind', 'block', 'sector',
    'phase', 'plot', 'flat', 'door', 'fl', 'no', 'unit', 'north', 'south', 'east', 'west',
    'india', 'us', 'usa', 'state', 'district', 'city', 'nagar', 'colony',
    'bazaar', 'marg', 'gali', 'null', 'rd', 'st', 'ave', 'blvd', 'dr', 'ct', 'ln',
    'hwy', 'apt', 'ste', 'first', 'second', 'third', 'ground',
    # French
    'france', 'rue', 'r', 'avenue', 'av', 'boulevard', 'bd', 'chemin', 'ch',
    'impasse', 'imp', 'allee', 'all', 'route', 'rte', 'cours', 'crs', 'quai', 'qu',
    'place', 'pl', 'square', 'sq', 'voie', 'passage', 'pass', 'cedex', 'bp',
    'boite', 'postale', 'etage', 'batiment', 'bat', 'immeuble', 'residence',
    'res', 'lieu', 'dit', 'lieudit', 'bis', 'ter', 'quater', 'zone', 'industrielle',
    'zi', 'za', 'activite', 'de', 'du', 'des', 'la', 'le', 'les', 'en', 'au', 'aux',
    'sur', 'sous', 'saint', 'sainte', 'd', 'l'
}


def get_blocking_keys_from_preprocessed(c_n, core_n, c_a, nums):
    """
    Generates blocking keys from pre-normalized fields without redundant string parsing.
    """
    keys = set()
    n_tokens = core_n.split()
    a_tokens = c_a.split()

    # 1. Compact name (no spaces)
    compact_name = ''.join(n_tokens)
    if len(compact_name) >= 4:
        keys.add(('compact_n', compact_name))
        keys.add(('compact_sort_n', ''.join(sorted(n_tokens))))

    # 2. Core name exact
    if len(core_n) >= 3:
        keys.add(('core_n', core_n))

    # 3. First 2 tokens of core name & sorted tokens
    if len(n_tokens) >= 2:
        keys.add(('n2', f'{n_tokens[0]}_{n_tokens[1]}'))
        keys.add(('sort_n', ' '.join(sorted(n_tokens))))
    elif len(n_tokens) == 1 and len(n_tokens[0]) >= 3:
        keys.add(('n1', n_tokens[0]))

    # 4. Individual significant name tokens
    for t in n_tokens:
        if len(t) >= 3 and t not in norm.LEGAL_SUFFIXES and t not in norm.ARTICLES_AND_PREP:
            keys.add(('n_tok', t))

    # 4b. Prefix n-grams (3, 4, 5-gram) for core name
    if len(core_n) >= 3:
        keys.add(('gram3', core_n[:3]))
    if len(core_n) >= 4:
        keys.add(('gram4', core_n[:4]))
    if len(core_n) >= 5:
        keys.add(('gram5', core_n[:5]))

    # 4c. Consonant skeleton & Soundex key (strips vowels to catch phonetic/spelling variations)
    vowels = set('aeiouy')
    cons_skel = ''.join([c for c in core_n if c.isalpha() and c not in vowels])
    if len(cons_skel) >= 4:
        keys.add(('cons_skel', cons_skel[:6]))

    if n_tokens:
        t0 = n_tokens[0]
        if len(t0) >= 3:
            snd = t0[0] + ''.join([c for c in t0[1:] if c.isalpha() and c not in 'aeiouy'])
            if len(snd) >= 3:
                keys.add(('snd', snd[:4]))

    if len(n_tokens) >= 2:
        if len(n_tokens[0]) >= 1:
            keys.add(('init_n2', f'{n_tokens[0][0]}_{n_tokens[1]}'))
        keys.add(('n_last', n_tokens[-1]))
        keys.add(('init_last', f'{n_tokens[0][0]}_{n_tokens[-1]}'))

    # Address tokens: extract significant words
    sig_addr_words = [t for t in a_tokens if t not in COMMON_ADDR_WORDS and len(t) >= 3 and not t.isdigit()]

    # 5. Number + City (last 2 tokens of address)
    if nums and len(a_tokens) >= 1:
        for num in nums:
            if len(num) >= 1:
                keys.add(('num_city1', f'{num}_{a_tokens[-1]}'))
                if len(a_tokens) >= 2:
                    keys.add(('num_city2', f'{num}_{a_tokens[-2]}'))

    # 6. Street number + rare address word
    if nums and sig_addr_words:
        for num in nums:
            for w in sig_addr_words[:3]:
                keys.add(('num_street', f'{num}_{w}'))

    # 7. Name token + Street number
    if n_tokens and nums:
        for num in nums:
            keys.add(('name_num', f'{n_tokens[0]}_{num}'))
            if len(n_tokens) >= 2:
                keys.add(('name_num2', f'{n_tokens[1]}_{num}'))

    # 8. Name token + City
    if n_tokens and len(a_tokens) >= 1:
        keys.add(('name_city', f'{n_tokens[0]}_{a_tokens[-1]}'))
        if len(a_tokens) >= 2:
            keys.add(('name_city2', f'{n_tokens[0]}_{a_tokens[-2]}'))

    # 8b. Significant Name Token + Significant Address Word
    if n_tokens and sig_addr_words:
        keys.add(('name_addr', f'{n_tokens[0]}_{sig_addr_words[0]}'))

    # 8c. Soundex of First Name + First Significant Address Word (Typo Tolerant)
    if n_tokens and sig_addr_words:
        t0 = n_tokens[0]
        if len(t0) >= 3:
            snd = t0[0] + ''.join([c for c in t0[1:] if c.isalpha() and c not in 'aeiouy'])
            if len(snd) >= 3:
                keys.add(('snd_addr', f'{snd[:4]}_{sig_addr_words[0]}'))

    # 9. Pairs of rare address words
    if len(sig_addr_words) >= 2:
        for i in range(min(len(sig_addr_words), 4)):
            for j in range(i + 1, min(len(sig_addr_words), 4)):
                w1, w2 = sorted([sig_addr_words[i], sig_addr_words[j]])
                keys.add(('addr_pair', f'{w1}_{w2}'))

    return keys


def get_blocking_keys(name, addr, country=None):
    """
    Generates multi-attribute blocking keys from business name and address strings.
    """
    c_n, core_n, _ = norm.normalize_name(name)
    c_a, nums, _, _ = norm.normalize_address(addr)
    return get_blocking_keys_from_preprocessed(c_n, core_n, c_a, nums)



def retrieve_candidates_for_s1(s1_records, inverted_index, top_k=50):
    """
    s1_records: dict of sid -> (name, addr, country)
    inverted_index: dict of key -> list of tids
    Returns: dict of sid -> list of (tid, shared_count)
    """
    results = {}
    for sid, (rname, raddr, rcountry) in s1_records.items():
        skeys = get_blocking_keys(rname, raddr, rcountry)
        counts = collections.Counter()
        for k in skeys:
            if k in inverted_index:
                counts.update(inverted_index[k])
        if counts:
            results[sid] = counts.most_common(top_k)
        else:
            results[sid] = []
    return results
