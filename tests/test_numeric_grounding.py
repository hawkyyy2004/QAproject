"""Text answers may only cite numbers the evidence itself states."""
import pytest

from qaproject.pipeline import NUMERIC_FLAG, numbers_in, numeric_flags, same_number
from test_bilingual import pair, run, setup_pipeline


def text_unit(text='海冰试样尺寸为7.5×7.5×17.0 cm，平均粒径0.6 cm，最大冰厚43 cm。'):
    return {'id': 'u1', 'text': text, 'image': '', 'language': 'zh', 'evidence_kind': 'extracted_text'}


def image_unit(text='图12 N2 与 30N2 峰面积的关系'):
    return {'id': 'u1', 'text': text, 'image': '/assets/a.png', 'language': 'zh', 'evidence_kind': 'pdf_figure'}


def versions(output):
    return [{'candidate_id': 'c1', 'qa': {'language': 'zh', 'output': output}}]


def test_identifiers_are_not_read_as_measurements():
    # N2/KJ12/u4 are names; the 30 in 30N2 is a mass number, a real quantity.
    assert numbers_in('N2、30N2、cm^3、KJ12 站和 u4') == [30]
    assert numbers_in('²⁸N2 与 ³0N2') == [28, 30]
    # A comma run is read both as a list and as a thousands mark; the evidence
    # decides which reading is real, so both are kept.
    assert {15, 34.5, 2000, 1250} <= set(numbers_in('温度 15 摄氏度，盐度34.5，深度 2000 m，面积 1,250 km^2'))


def test_fullwidth_digits_compare_like_ascii():
    assert numbers_in('盐度为３４.５') == [34.5]


@pytest.mark.parametrize('left,right', [
    (1.767, 1.77),        # rounded for reporting
    (15, 15.0),
    (0.844, 84.4),        # percent written as a fraction
    (84.4, 0.844),
])
def test_equivalent_values_are_accepted(left, right):
    assert same_number(__import__('decimal').Decimal(str(left)), __import__('decimal').Decimal(str(right)))


@pytest.mark.parametrize('left,right', [
    (5, 5.5),             # different values, not a rounding difference
    (221, 22),
    (0.32, 0.23),
])
def test_different_values_are_rejected(left, right):
    assert not same_number(__import__('decimal').Decimal(str(left)), __import__('decimal').Decimal(str(right)))


def test_answer_numbers_must_appear_in_the_evidence():
    assert numeric_flags(versions('最大冰厚43 cm，试样尺寸7.5×7.5×17.0 cm。'), text_unit()) == []
    flagged = numeric_flags(versions('最大冰厚43 cm，侧限板间距为12.5 cm。'), text_unit())
    assert flagged and NUMERIC_FLAG in flagged[0] and '12.5' in flagged[0]


def test_image_evidence_is_exempt_because_readings_come_from_the_crop():
    assert numeric_flags(versions('约 6.2 μW/(cm²·nm·sr)'), image_unit()) == []


def test_evidence_without_numbers_cannot_vouch_for_any():
    assert numeric_flags(versions('温度 15 摄氏度。'), text_unit('该站为咸淡水混合区。')) == [
        f'{NUMERIC_FLAG}：15。请核对是否写入了原文以外的数值，或改用原文可核实的表述。']


def test_fabricated_number_blocks_the_candidate_and_triggers_rewrite(setup_pipeline):
    _, store, _, _, calls, behavior, _ = setup_pipeline
    fabricated = []

    def invented_number(result, payload, units):
        # Only the first draft lies; the rewrite the veto triggers is honest.
        if not payload.get('repair') and not fabricated:
            fabricated.append(1)
            for q in result['pairs']:
                q['zh']['output'] = '温度为 27.4 摄氏度。'
                q['en']['output'] = 'The temperature was 27.4 degrees Celsius.'
        return result

    behavior['generation'] = invented_number
    doc = run(setup_pipeline)
    assert doc['status'] == 'waiting_review'
    assert doc['bilingual_count'] == 1
    assert not any(NUMERIC_FLAG in str(c['review'].get('local_flags', [])) for c in doc['candidates'] if c['status']!='discarded')
    # The veto reaches the model: a rewrite is attempted, not a silent re-review.
    assert any('repair' in payload for kind, payload, _ in calls if kind == 'generation')
    assert any('27.4' in str(payload.get('repair', {})) for kind, payload, _ in calls
               if kind == 'generation')


def test_unfixed_fabrication_never_reaches_the_library(setup_pipeline):
    _, store, _, _, _, behavior, _ = setup_pipeline

    def always_invent(result, payload, units):
        for q in result['pairs']:
            q['zh']['output'] = '温度为 27.4 摄氏度。'
            q['en']['output'] = 'The temperature was 27.4 degrees Celsius.'
        return result

    behavior['generation'] = always_invent
    doc = run(setup_pipeline)
    assert doc['status'] == 'insufficient' and doc['qa_count'] == 0
    assert all(c['status'] != 'pending' for c in doc['candidates'])
