"""MSP (15% span corruption) and NTP (10–90% random pivot), paper §3.2."""

import math


def ntp_pair(ids, rng):
    if len(ids) < 2:
        raise ValueError("NTP requires at least two tokens")
    pivot = rng.randint(max(1, math.ceil(len(ids) * 0.1)), min(len(ids) - 1, math.floor(len(ids) * 0.9)))
    return ids[:pivot], ids[pivot:]


def msp_pair(ids, rng, sentinels, corruption=0.15, mean_span=3):
    if len(ids) < 2 or not 0 < corruption < 1:
        raise ValueError("Invalid span corruption input")
    count = min(len(ids) - 1, max(1, round(len(ids) * corruption)))
    positions = set()
    # Sample disjoint runs until exactly the requested corruption budget is met.
    while len(positions) < count:
        start = rng.randrange(len(ids))
        size = max(1, round(rng.expovariate(1 / mean_span)))
        for i in range(start, min(start + size, len(ids))):
            positions.add(i)
            if len(positions) == count:
                break
    source, target = [], []
    span = -1
    for i, token in enumerate(ids):
        if i in positions:
            if i - 1 not in positions:
                span += 1
                if span + 1 >= len(sentinels):
                    raise ValueError("Not enough sentinel tokens for this corruption")
                source.append(sentinels[span])
                target.append(sentinels[span])
            target.append(token)
        else:
            source.append(token)
    target.append(sentinels[span + 1])
    return source, target
