"""Figure questions get prose only when the prose names that figure."""
from qaproject.glm import BLIND, GENERATE, REVIEW, VISUAL_GUIDANCE
from qaproject.pipeline import blind_passes, cited_refs, figure_key, visual_background


def unit(uid, page, text='', *, kind='extracted_text', image=False, **extra):
    return dict(id=uid, page=page, text=text, evidence_kind=kind, locator=f'page-{page:02d}-seg-01',
                language='zh', image=(f'/assets/{uid}.png' if image else ''), **extra)


def test_figure_key_reads_the_caption_not_the_prose():
    table = unit('t1', 6, '表16个典型变量与显著性检验结果 Tab. 1 Summary of 16 variables', kind='pdf_table')
    assert figure_key(table) == ('table', '1')
    assert figure_key(unit('t2', 6, '表2 相关系数矩阵', kind='pdf_table')) == ('table', '2')
    assert figure_key(unit('f4', 4, '图 4 台风路径 Figure 4 Track', kind='pdf_figure')) == ('figure', '4')
    assert figure_key(unit('f9', 9, '未命名插图', kind='pdf_figure')) is None


def test_cited_refs_collects_figure_and_table_references_separately():
    assert cited_refs('如图4和图5所示，表2给出显著性') == {
        ('figure', '4'), ('figure', '5'), ('table', '2')}
    assert cited_refs('Fig. 3 and Table 12 compare the treatments') == {
        ('figure', '3'), ('table', '12')}
    assert cited_refs('没有点名任何图表') == set()


def test_figure_and_table_sharing_a_number_are_not_confused():
    fig1 = unit('f1', 6, '图1 站位分布', kind='pdf_figure', image=True)
    tbl1 = unit('t1', 6, '表1 站位坐标 Tab. 1 Station coordinates', kind='pdf_table', image=True)
    prose = unit('p1', 6, '如图1所示，站位覆盖整个海湾。')
    units = [fig1, tbl1, prose]

    assert visual_background(units, [fig1]) == prose['text']
    assert visual_background(units, [tbl1]) == ''


def test_prose_naming_two_figures_is_shared_once_and_unnamed_figure_gets_none():
    prose = unit('p1', 4, '图4和图5给出两种处理下的溶解氧剖面，随深度变化的趋势相反。')
    fig4 = unit('f4', 4, '图4 处理A的溶解氧剖面', kind='pdf_figure', image=True)
    fig5 = unit('f5', 4, '图5 处理B的溶解氧剖面', kind='pdf_figure', image=True)
    fig6 = unit('f6', 4, '图6 风速时间序列', kind='pdf_figure', image=True)
    units = [prose, fig4, fig5, fig6]

    assert visual_background(units, [fig4]) == prose['text']
    assert visual_background(units, [fig5]) == prose['text']
    # Nothing names figure 6, so it must not borrow its neighbours' paragraph.
    assert visual_background(units, [fig6]) == ''
    assert visual_background(units, [fig4, fig5]) == prose['text']


def test_unnamed_prose_goes_to_the_nearest_figure_only():
    fig_a = unit('fa', 3, '图1 温盐剖面', kind='pdf_figure', image=True)
    prose = unit('p1', 3, '观测在春季完成，站位水深超过两千米。')
    fig_b = unit('fb', 3, '图2 流速剖面', kind='pdf_figure', image=True)
    units = [fig_a, prose, fig_b]

    assert visual_background(units, [fig_a]) == prose['text']
    assert visual_background(units, [fig_b]) == ''


def test_text_copy_of_a_figure_is_not_prose_for_its_neighbour():
    fig1 = unit('f1', 2, '图1 叶绿素剖面', kind='pdf_figure', image=True)
    copy = unit('f1:text', 2, '图1 叶绿素剖面 0 10 20 30 40', companion_of='f1')
    fig2 = unit('f2', 2, '图2 营养盐剖面', kind='pdf_figure', image=True)
    units = [fig1, copy, fig2]

    assert visual_background(units, [fig1]) == ''
    assert visual_background(units, [fig2]) == ''


def test_whole_page_render_never_receives_background():
    prose = unit('p1', 1, '图1 站位分布')
    page = unit('pg', 1, '整页扫描的文字层', kind='pdf_page', image=True)
    assert visual_background([prose, page], [page]) == ''


def test_blind_verdict_blocks_only_explicit_disagreement():
    assert blind_passes(None)
    assert blind_passes({'consistent': True})
    assert not blind_passes({'consistent': False})
    assert not blind_passes({})


def test_generator_and_reviewer_share_the_visual_standard():
    assert VISUAL_GUIDANCE in GENERATE
    assert '优先考查图表传达的科学结论' in GENERATE
    assert '看不见文字说明' in BLIND and 'consistent' in BLIND
    assert '只要求读出曲线、序列或表中的某一数值' in REVIEW
