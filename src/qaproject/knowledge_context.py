"""Local, conservative naming and prompt context; never adds scientific facts."""
import re


def explicit_aliases(units):
    """Only definitions with matching initials or an explicit Chinese alias marker.

    Conflicting definitions stay unresolved. No similarity or cross-file lookup.
    """
    candidates = {}
    def add(alias, full):
        if alias.lower() != full.lower():
            candidates.setdefault(alias.lower(), set()).add(full)
    for unit in units:
        text = unit.get('text', '')
        for match in re.finditer(r'\b([A-Za-z][A-Za-z -]{3,100})\s*\(([A-Z][A-Z0-9-]{1,11})\)', text):
            words = match[1].strip().split()
            short = match[2]
            # Take the shortest suffix whose initials match, avoiding preceding prose.
            for n in range(2, min(len(words), 12)+1):
                full = ' '.join(words[-n:])
                initials = ''.join(w[0] for w in words[-n:]).upper()
                if initials == short.replace('-', ''):
                    add(short, full); break
        for match in re.finditer(r'([\u4e00-\u9fff]{2,30})[（(](?:以下简称|简称|又称)[“"\s]*([\u4e00-\u9fffA-Za-z0-9-]{2,20})[”"\s]*[）)]', text):
            add(match[2], match[1])
    return {alias: next(iter(names)) for alias, names in candidates.items() if len(names) == 1}


def canonical_name(name, aliases, existing, entity_type):
    """Prefer the existing reviewed spelling, only within a proven alias pair."""
    full = aliases.get(name.lower())
    if full is None:
        full = next((v for v in aliases.values() if v.lower() == name.lower()), None)
    if full is None:
        return name
    family = {full.lower(), *(k for k,v in aliases.items() if v.lower() == full.lower())}
    for row in existing:
        if row['entity_type'] == entity_type and row['entity_name'].lower() in family:
            return row['entity_name']
    return full


def entity_context(existing, units, names=(), aliases=None):
    """Full facts for named entities; a compact identity index for everything else."""
    aliases = aliases or {}
    text = '\n'.join(u.get('text', '') for u in units).lower()
    selected = {n.lower() for n in names}
    for row in existing:
        name = row['entity_name'].lower()
        if name in text:
            selected.add(name)
    for alias, full in aliases.items():
        if alias in selected or full.lower() in selected or alias in text or full.lower() in text:
            selected.update((alias, full.lower()))
    related, index, seen = [], [], set()
    for row in existing:
        key = (row['entity_name'].lower(), row['entity_type'])
        if key[0] in selected:
            related.append(row)
        elif key not in seen:
            index.append(dict(entity_name=row['entity_name'], entity_type=row['entity_type']))
        seen.add(key)
    return related, index
