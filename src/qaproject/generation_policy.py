"""Plan evidence coverage locally, without another selection/model request."""
import re


def generation_limits(config):
    """Text and image quotas are separate so neither starves the other."""
    target = max(1, min(10, int(config['target_per_document'])))
    # Small explicit targets remain useful for isolated validation and deployments.
    if target < 5:
        return dict(target=target, maximum=target, text=target, visual=target)
    maximum = max(target, min(10, int(config.get('max_per_document', target))))
    text = min(5, target)
    visual = min(5, maximum - text)
    return dict(target=target, maximum=maximum, text=text, visual=visual)


def spread(groups):
    """Interleave head and tail so the last pages are reached before quota runs out.

    Evidence arrives in document order and a phase used to stop as soon as the
    quota was met, so the closing chapters never got a single question.
    """
    ordered, left, right = [], 0, len(groups) - 1
    while left <= right:
        ordered.append(groups[left])
        left += 1
        if left <= right:
            ordered.append(groups[right])
            right -= 1
    return ordered


def phase_quota(goal, groups):
    """Largest share one evidence group may take before the next one gets a turn."""
    return max(1, -(-goal // max(1, len(groups))))


def visual_quota(visual_groups, limits):
    """Two questions per figure at most; a single figure cannot fill a five-image target."""
    return min(limits['visual'], 2*len(visual_groups)) if visual_groups else 0


def generation_phases(text_groups, visual_groups, limits):
    if text_groups:
        yield 'text', min(3, limits['text']) if visual_groups else limits['text'], False, spread(text_groups)
    if visual_groups:
        yield 'visual', visual_quota(visual_groups, limits), False, spread(visual_groups)
    if text_groups and visual_groups:
        # A missing/unusable figure is replaced with text, never a fabricated visual QA.
        yield 'text', limits['text'], True, spread(text_groups)


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
