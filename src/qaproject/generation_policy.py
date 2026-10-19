"""Plan evidence coverage locally, without another selection/model request."""
import re


def generation_limits(config):
    target = max(1, min(7, int(config['target_per_document'])))
    # Small explicit targets remain useful for isolated validation and deployments.
    maximum = target if target < 5 else max(target, min(7, int(config.get('max_per_document', 7))))
    return dict(target=target, maximum=maximum, text=min(5, target), visual=min(2, maximum))


def generation_phases(text_groups, visual_groups, limits):
    if text_groups:
        yield 'text', min(3, limits['text']) if visual_groups else limits['text'], False, text_groups
    if visual_groups:
        yield 'visual', limits['visual'], False, visual_groups
    if text_groups and visual_groups:
        # A missing/unusable figure is replaced with text, never a fabricated visual QA.
        yield 'text', limits['text'], True, text_groups


def modality_counts(candidates):
    pairs = {}
    for c in candidates:
        if c.get('pair_id') and c['status'] not in ('rejected', 'discarded'):
            pairs.setdefault(c['pair_id'], []).append(c)
    counts = dict(text=0, visual=0)
    for versions in pairs.values():
        if len(versions) == 2 and {c['row']['language'] for c in versions} == {'zh', 'en'}:
            counts['visual' if versions[0]['row']['image'] else 'text'] += 1
    return counts


def visual_priority(unit):
    # Prefer existing figure/table captions; no OCR, image decode or PDF rereading.
    return -len(re.findall(r'(?:图|表|\bfig(?:ure)?\.?|\btable)\s*\d+', unit['text'], re.I))
