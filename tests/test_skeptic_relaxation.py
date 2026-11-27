"""The skeptic may veto facts, not taste; value objections stay advisory."""
from qaproject.glm import GLMClient, REVIEW_SKEPTIC
from test_bilingual import run, setup_pipeline


def override_skeptic(monkeypatch, **flags):
    base = GLMClient.call

    def call(self, instruction, payload, units=(), **kwargs):
        result, meta = base(self, instruction, payload, units, **kwargs)
        if instruction.startswith(REVIEW_SKEPTIC[:24]) and 'checks' in result:
            for check in result['checks']:
                check.update(flags)
        return result, meta

    monkeypatch.setattr(GLMClient, 'call', call)


def test_value_and_novelty_objections_do_not_block(setup_pipeline, monkeypatch):
    """knowledge_value/distinct are the skeptic's opinion; the pair already passed."""
    override_skeptic(monkeypatch, knowledge_value=False, distinct=False, topic_correct=False)
    doc = run(setup_pipeline)
    assert doc['status'] == 'waiting_review'
    assert doc['bilingual_count'] == 3
    assert all(c['status'] == 'pending' for c in doc['candidates'])
    notes = [c['review']['note'] for c in doc['candidates']]
    assert any('怀疑者提示（不阻断）' in note for note in notes)


def test_factual_objection_still_blocks(setup_pipeline, monkeypatch):
    """A skeptic that disputes support or scope keeps its veto."""
    override_skeptic(monkeypatch, supported=False)
    doc = run(setup_pipeline)
    assert doc['status'] == 'insufficient' and doc['qa_count'] == 0
