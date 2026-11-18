# Картка корпусу (Corpus Card) — ЛР 10

**Предметна область:** «Управління надзвичайними ситуаціями» (тема №30, проєкт Domain Knowledge Hub).
**Дата збирання:** 2026-11-18 · **Скрипт:** `src/build_corpus.py` (відтворюваний, сирі відповіді API кешуються в `data/raw/wiki/`).

## 1. Підсумок
| Показник | Значення |
|---|---|
| Документів | 50 |
| Фрагментів (стратегія `structure`, файл `data/corpus.jsonl`) | 849 |
| Фрагментів за стратегіями | fixed: 896, structure: 849, hierarchical: 2166 |
| Слів після очищення | 90820 |
| Мови документів | uk: 46, en: 4 |
| Типи документів | article: 30, record: 20 |
| Фрагменти за sensitivity | public: 674, restricted: 175 |

## 2. Джерела та ліцензії
| Ліцензія / умови | Документів |
|---|---|
| CC BY-SA 4.0 (Вікіпедія) | 30 |
| EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення | 20 |

- **Вікіпедія (uk, en)** — тексти статей отримано через MediaWiki API (`prop=extracts`, plain text). Ліцензія CC BY-SA 4.0: дозволено використання з зазначенням авторства й посиланням (`source_url` у кожному фрагменті).
- **EM-DAT** (події 2023 р.; CRED / UCLouvain) — дані надано для некомерційного використання після реєстрації; повторне поширення сирих даних заборонене, тому фрагменти позначено `restricted` і не публікуються поза репозиторієм курсу.
- Нотатки kb/*.md з ЛР 2 не використано (відсутні).

## 3. Метадані фрагмента
`chunk_id, doc_id, title, section, type, date, sensitivity, source_url, lang, text, n_words, units, strategy, context_id`
(`date` — дата останньої редакції статті / дата початку події EM-DAT / дата зміни файлу kb; `units` — розділи документа, які покриває фрагмент, використовуються для розмітки релевантності).

## 4. Правила очищення від шуму
1. Беремо лише текст статті (`explaintext`) — меню, навігаційні шаблони, інфобокси, категорії й колонтитули сторінки відкидаються API.
2. Видаляються службові розділи разом із підрозділами: «Примітки», «Джерела», «Посилання», «Література», «Див. також», «Галерея» (та en-відповідники).
3. Видаляються виноски `[12]`, позначки `[джерело?]`, `[en]`, залишки формул `{\displaystyle…}`, стрілки `↑`.
4. Рядки коротші за 3 слова (підписи, залишки списків-навігації) відкидаються; пробіли нормалізуються.
5. Повторювані абзаци в межах документа видаляються.
6. Документи, коротші за 150 слів після очищення, відкидаються.

**Статистика застосування правил:**
| Правило | Спрацювань |
|---|---|
| короткі рядки (<3 слів: підписи, навігація) | 210 |
| службові розділи (Примітки, Посилання, Див. також…) | 97 |
| порожні після очищення розділи | 48 |
| залишки формул {\displaystyle} | 20 |
| службові позначки [джерело?]/[en] | 5 |
| посилання-виноски [n] | 3 |
| документи < min_words (відкинуто) | 1 |
| стрілки виносок ↑ | 0 |

## 5. Обмеження доступу (sensitivity)
| Рівень | Правило | Хто бачить |
|---|---|---|
| `public` | відкриті ліцензії (Вікіпедія) | усі |
| `internal` | власні нотатки команди (kb з ЛР 2) | учасники проєкту |
| `restricted` | дані з ліцензійними обмеженнями (EM-DAT) | лише авторизовані аналітики; виключаються фільтром для ролі `public` |

## 6. Документи
| doc_id | Назва | Тип | Мова | Sensitivity | Дата | Розділів | Слів | Фрагм. | Ліцензія |
|---|---|---|---|---|---|---|---|---|---|
| wiki_uk_01 | Надзвичайна ситуація | article | uk | public | 2025-08-15 | 22 | 3457 | 26 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_02 | Цивільна оборона | article | uk | public | 2026-08-18 | 5 | 830 | 6 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_03 | Державна служба України з надзвичайних ситуацій | article | uk | public | 2026-11-12 | 25 | 2990 | 25 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_04 | Стихійне лихо | article | uk | public | 2026-11-04 | 6 | 2259 | 13 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_05 | Землетрус | article | uk | public | 2026-11-10 | 17 | 2177 | 18 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_06 | Цунамі | article | uk | public | 2026-09-28 | 11 | 2014 | 14 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_07 | Повінь | article | uk | public | 2025-09-14 | 3 | 768 | 5 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_08 | Тропічний циклон | article | uk | public | 2026-11-15 | 37 | 6598 | 51 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_09 | Смерч | article | uk | public | 2026-09-12 | 17 | 3173 | 24 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_10 | Посуха | article | uk | public | 2026-06-03 | 9 | 571 | 6 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_11 | Лісова пожежа | article | uk | public | 2025-09-02 | 7 | 1376 | 11 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_12 | Виверження вулкана | article | uk | public | 2026-11-03 | 7 | 391 | 4 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_13 | Зсув | article | uk | public | 2025-10-19 | 1 | 158 | 1 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_14 | Лавина | article | uk | public | 2026-09-20 | 5 | 897 | 7 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_15 | Хвилі спеки в Європі (2026) | article | uk | public | 2026-10-27 | 93 | 14984 | 122 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_16 | Епідемія | article | uk | public | 2026-11-05 | 8 | 965 | 9 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_17 | Пандемія COVID-19 | article | uk | public | 2026-05-04 | 65 | 7547 | 62 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_18 | Чорнобильська катастрофа | article | uk | public | 2026-11-14 | 35 | 8504 | 58 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_19 | Підрив Каховської ГЕС | article | uk | public | 2026-11-08 | 19 | 3740 | 27 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_20 | Міжнародна реакція на землетрус у Туреччині та Сирії 2023 року | article | uk | public | 2026-11-14 | 108 | 6140 | 57 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_21 | Землетрус в Індійському океані (2004) | article | uk | public | 2026-11-08 | 4 | 659 | 5 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_22 | Землетрус у Японії (2011) | article | uk | public | 2026-08-10 | 5 | 466 | 5 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_23 | Катріна (ураган) | article | uk | public | 2026-06-15 | 8 | 581 | 7 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_24 | Евакуація | article | uk | public | 2026-02-03 | 1 | 155 | 1 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_25 | Радіаційна аварія | article | uk | public | 2025-08-05 | 3 | 368 | 3 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_uk_26 | Медицина катастроф | article | uk | public | 2026-11-06 | 7 | 905 | 6 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_en_01 | Emergency management | article | en | public | 2026-11-15 | 42 | 6885 | 52 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_en_02 | Sendai Framework for Disaster Risk Reduction | article | en | public | 2026-06-28 | 5 | 429 | 5 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_en_03 | Incident Command System | article | en | public | 2026-10-20 | 28 | 4887 | 38 | CC BY-SA 4.0 (Вікіпедія) |
| wiki_en_04 | Early warning system | article | en | public | 2026-08-31 | 9 | 737 | 6 | CC BY-SA 4.0 (Вікіпедія) |
| emdat_air | EM-DAT 2023: Air | record | uk | restricted | 2023-09-16 | 4 | 80 | 4 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_collapse_industrial_ | EM-DAT 2023: Collapse (Industrial) | record | uk | restricted | 2023-12-01 | 6 | 170 | 6 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_collapse_miscellaneous_ | EM-DAT 2023: Collapse (Miscellaneous) | record | uk | restricted | 2023-07-23 | 4 | 97 | 4 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_drought | EM-DAT 2023: Drought | record | uk | restricted | 2023-09 | 10 | 333 | 10 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_earthquake | EM-DAT 2023: Earthquake | record | uk | restricted | 2023-12-19 | 12 | 401 | 12 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_epidemic | EM-DAT 2023: Epidemic | record | uk | restricted | 2023-12-06 | 12 | 401 | 12 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_explosion_industrial_ | EM-DAT 2023: Explosion (Industrial) | record | uk | restricted | 2023-12-24 | 11 | 300 | 11 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_explosion_miscellaneous_ | EM-DAT 2023: Explosion (Miscellaneous) | record | uk | restricted | 2023-08-14 | 3 | 82 | 3 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_extreme_temperature | EM-DAT 2023: Extreme temperature | record | uk | restricted | 2023-11-19 | 10 | 286 | 10 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_fire_industrial_ | EM-DAT 2023: Fire (Industrial) | record | uk | restricted | 2023-09-24 | 3 | 77 | 3 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_fire_miscellaneous_ | EM-DAT 2023: Fire (Miscellaneous) | record | uk | restricted | 2023-11-16 | 12 | 307 | 12 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_flood | EM-DAT 2023: Flood | record | uk | restricted | 2023-10-15 | 12 | 437 | 12 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_mass_movement_wet_ | EM-DAT 2023: Mass movement (wet) | record | uk | restricted | 2023-12-25 | 12 | 368 | 12 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_miscellaneous_accident_general_ | EM-DAT 2023: Miscellaneous accident (General) | record | uk | restricted | 2023-11-20 | 9 | 236 | 9 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_rail | EM-DAT 2023: Rail | record | uk | restricted | 2023-12-14 | 3 | 63 | 3 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_road | EM-DAT 2023: Road | record | uk | restricted | 2023-12-26 | 12 | 253 | 12 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_storm | EM-DAT 2023: Storm | record | uk | restricted | 2023-11-26 | 12 | 507 | 12 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_volcanic_activity | EM-DAT 2023: Volcanic activity | record | uk | restricted | 2023-12-03 | 4 | 131 | 4 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_water | EM-DAT 2023: Water | record | uk | restricted | 2023-12-16 | 12 | 235 | 12 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |
| emdat_wildfire | EM-DAT 2023: Wildfire | record | uk | restricted | 2023-11-17 | 12 | 445 | 12 | EM-DAT, CRED/UCLouvain — некомерційне використання, без поширення |

## 7. Відомі прогалини та упередження
- Вікіпедія — енциклопедичне джерело, а не нормативне: немає чинних регламентів ДСНС, планів реагування, наказів; деякі статті uk-Вікіпедії короткі або застарілі.
- З EM-DAT (події 2023 р.) взято лише топ-12 подій кожного типу за кількістю загиблих — дрібні події не представлені.
- Пошук Вікіпедії для кількох тем повернув вужчі статті («Хвилі спеки в Європі (2026)», «Міжнародна реакція на землетрус у Туреччині та Сирії 2023 року»); вони великі й переважають за кількістю фрагментів; стаття «Сель» відкинута як закоротка.
- Корпус переважно україномовний; англомовних документів мало (вони потрібні для перевірки багатомовності моделей).
- `date` для статей — дата останньої редакції, а не дата події/публікації.
- Числові дані у статтях можуть розходитися з EM-DAT (різні методики підрахунку).
