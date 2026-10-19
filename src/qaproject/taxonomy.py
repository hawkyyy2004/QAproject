"""Select formal topics once per document, without another model request."""
from .store import normalized, limit_search_terms, search_term_key
from .settings import MIN_DOCUMENT_TOPICS, MAX_DOCUMENT_TOPICS


class TopicSelection:
    def __init__(self, known_topics, units):
        self.known = {normalized(name): t for t in known_topics for name in [t['name'], *t['aliases']]}
        self.units = {u['id']: u for u in units}
        positions = {u['id']: i for i, u in enumerate(units)}
        self.positions = {u['id']: positions.get(u.get('companion_of'), positions[u['id']]) for u in units}
        self.proposals = {}
        self.deferred = []
        self.extra_terms = {}

    def add(self, result, group):
        if not isinstance(result.get('summary'), str) or not isinstance(result.get('topics'), list):
            raise ValueError('主题分析响应结构无效。')
        allowed = {u['id'] for u in group}
        for t in result['topics']:
            if not isinstance(t, dict) or not isinstance(t.get('name'), str):
                raise ValueError('主题分析响应结构无效。')
            support = t.get('unit_ids', [])
            if not isinstance(support, list) or not support or any(not isinstance(x, str) for x in support) or not set(support) <= allowed:
                raise ValueError('主题缺少当前证据支持。')
            words = t.get('keywords', [])
            if not isinstance(words, list) or any(not isinstance(w, str) or not w.strip() or len(w) > 80 for w in words):
                raise ValueError('检索词结构无效。')
            importance = t.get('importance', 2)
            if type(importance) is not int or importance not in (1, 2, 3):
                raise ValueError('主题重要性必须为1至3。')
            existing = self.known.get(normalized(t['name']))
            name = existing['name'] if existing else t['name'].strip()
            definition, reason = t.get('definition', ''), t.get('new_reason', '')
            if not existing and (not name or len(name) > 24 or not isinstance(definition, str) or not definition.strip()
                                 or not isinstance(reason, str) or not reason.strip()):
                self.deferred.append(dict(name=name[:80], reason='新主题缺少规范名称、定义或独立分类理由，未登记正式主题。',
                                          unit_ids=list(support)))
                for word in [*words, *([name] if name and len(name) <= 80 else [])]:
                    self.extra_terms.setdefault(word, set()).update(support)
                continue
            key = normalized(name)
            proposal = self.proposals.setdefault(key, dict(name=name, definition=definition, new_reason=reason,
                id=existing['id'] if existing else None, keywords=set(), unit_ids=set(), locations=set(),
                importance=importance, first=len(self.units), order=len(self.proposals), terms={}))
            proposal['importance'] = max(proposal['importance'], importance)
            proposal['keywords'].update(words)
            for word in words:
                proposal['terms'].setdefault(word, set()).update(support)
            for unit_id in support:
                unit = self.units[unit_id]
                proposal['unit_ids'].add(unit_id)
                proposal['locations'].add(unit['locator'])
                proposal['first'] = min(proposal['first'], self.positions[unit_id])
                parent = self.units.get(unit.get('companion_of'))
                if parent and parent['text'] == unit['text'] and parent['locator'] == unit['locator']:
                    proposal['unit_ids'].add(parent['id'])

    def select(self, required_ids=()):
        required = set(required_ids)
        if len(required) > MAX_DOCUMENT_TOPICS:
            raise ValueError('历史候选引用超过5个主题，需另行审核归类迁移；未改动QA及审核记录。')
        ranked = sorted(self.proposals.values(), key=lambda t: (
            t['id'] not in required, -t['importance'], -len(t['locations']), not bool(t['id']), t['first'], t['order']))
        selected = ranked[:MAX_DOCUMENT_TOPICS]
        if not required <= {t['id'] for t in selected}:
            raise ValueError('历史候选的主题缺少本次证据支持，需重新核验归类。')
        if len(selected) < MIN_DOCUMENT_TOPICS:
            raise ValueError('未识别到有证据支持的正式主题，归类未完成，请重试或检查原文；不以占位主题补数。')
        # Search terms retain their own support; they never expand generation permissions.
        terms, priorities = {}, {}
        for index, proposal in enumerate(ranked):
            words = dict(proposal['terms'])
            if proposal not in selected:
                words[proposal['name']] = proposal['unit_ids']
            for word, support in words.items():
                terms.setdefault(word, set()).update(support)
                priority = (int(index >= MAX_DOCUMENT_TOPICS), -proposal['importance'])
                key = search_term_key(word)
                priorities[key] = min(priorities.get(key, priority), priority)
        for word, support in self.extra_terms.items():
            terms.setdefault(word, set()).update(support)
        terms = limit_search_terms(terms, limit=None)
        ordered = sorted(terms, key=lambda word: (
            priorities.get(search_term_key(word), (2, 0)),
            -len({self.units[u]['locator'] for u in terms[word]}),
            min(self.positions[u] for u in terms[word])))
        terms = limit_search_terms({word: terms[word] for word in ordered})
        retained = {search_term_key(word) for word in terms}
        selected = [dict(t, keywords={w for w in t['keywords'] if search_term_key(w) in retained}) for t in selected]
        return selected, terms
