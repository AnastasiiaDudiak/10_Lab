# Узгодженість розмітників (κ Коена)

Розмітник A — автоматичний (cross-encoder `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, score > 0 → 1); розмітник B — LLM (Claude Opus 5.5).
Запитів: 10; пар (запит, фрагмент): 92.

|  | B = 1 | B = 0 |
|---|---|---|
| A = 1 | 16 | 23 |
| A = 0 | 2 | 51 |

p_o (спостережувана згода) = 0.728; p_e (випадкова згода) = 0.546;
**κ = (p_o − p_e) / (1 − p_e) = 0.401** — помірна згода (шкала Landis & Koch, 1977).

## Розбіжності (25)
| qid | chunk_id | A | B |
|---|---|---|---|
| q01 | wiki_uk_09::hierarchical-parent::0009 | 1 | 0 |
| q03 | wiki_uk_18::hierarchical-parent::0005 | 0 | 1 |
| q03 | wiki_uk_18::hierarchical-parent::0014 | 0 | 1 |
| q05 | wiki_uk_19::hierarchical-parent::0002 | 1 | 0 |
| q05 | wiki_uk_19::hierarchical-parent::0012 | 1 | 0 |
| q05 | wiki_uk_19::hierarchical-parent::0013 | 1 | 0 |
| q05 | wiki_uk_19::hierarchical-parent::0014 | 1 | 0 |
| q06 | emdat_air::hierarchical::0000 | 1 | 0 |
| q06 | emdat_air::hierarchical::0003 | 1 | 0 |
| q06 | emdat_earthquake::hierarchical::0010 | 1 | 0 |
| q06 | emdat_rail::hierarchical::0001 | 1 | 0 |
| q06 | emdat_road::hierarchical::0010 | 1 | 0 |
| q06 | emdat_water::hierarchical::0002 | 1 | 0 |
| q06 | emdat_water::hierarchical::0007 | 1 | 0 |
| q06 | emdat_water::hierarchical::0010 | 1 | 0 |
| q06 | wiki_uk_20::hierarchical-parent::0015 | 1 | 0 |
| q06 | wiki_uk_20::hierarchical-parent::0019 | 1 | 0 |
| q06 | wiki_uk_20::hierarchical-parent::0049 | 1 | 0 |
| q08 | emdat_storm::hierarchical::0002 | 1 | 0 |
| q08 | emdat_storm::hierarchical::0005 | 1 | 0 |
| q08 | emdat_storm::hierarchical::0006 | 1 | 0 |
| q08 | emdat_storm::hierarchical::0007 | 1 | 0 |
| q08 | wiki_uk_20::hierarchical-parent::0028 | 1 | 0 |
| q09 | wiki_uk_26::hierarchical-parent::0000 | 1 | 0 |
| q10 | wiki_uk_03::hierarchical-parent::0014 | 1 | 0 |
