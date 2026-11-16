"""A figure question must ask for a judgement, not list what the picture contains."""
import pytest

from qaproject.pipeline import VISUAL_FLAG, visual_question_flags


def versions(question, language='zh'):
    return [{'candidate_id': 'c1', 'qa': {'language': language, 'input': question, 'output': '答案'}}]


FIGURE = {'id': 'u1', 'image': '/assets/page-05-fig-01.png', 'evidence_kind': 'pdf_figure'}


@pytest.mark.parametrize('question', [
    '在系统体系结构图中，数据库层包含哪几个数据库？',
    '三类用户分别采用什么体系结构？',
    '图中水深最大值是多少？',
    '该图包括哪些要素？',
])
def test_enumeration_shapes_are_flagged(question):
    flags = visual_question_flags(versions(question), FIGURE)
    assert flags and VISUAL_FLAG in flags[0]


@pytest.mark.parametrize('question', [
    '专业层用户与系统维护层用户在权限和体系结构上有何不同？为什么这样划分？',
    '三维地层体采用半透明渲染，这对检查建模结果有什么作用？',
    '浊度峰值出现在表层以下，这一现象对判断悬浮物来源意味着什么？',
])
def test_interpretive_questions_pass(question):
    assert visual_question_flags(versions(question), FIGURE) == []


def test_text_questions_are_not_judged_on_figure_shape():
    assert visual_question_flags(versions('数据库中包含哪几个数据库？'), {'id': 'u2', 'image': ''}) == []
